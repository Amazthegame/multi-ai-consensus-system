import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import anthropic
from openai import OpenAI
from google import genai
import requests


class ChatHistoryExtractor:
    def __init__(self,
                 claude_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 gemini_api_key: Optional[str] = None,
                 perplexity_api_key: Optional[str] = None):
        """
        Initialize the chat history extractor with optional API keys.
        If a key is not provided, that platform will be skipped.
        """
        self.claude_client = None
        self.openai_client = None
        self.gemini_client = None
        self.perplexity_api_key = None
        self.active_platforms = []

        if claude_api_key:
            try:
                self.claude_client = anthropic.Anthropic(api_key=claude_api_key)
                self.active_platforms.append('claude')
                print("✓ Claude API configured (with web search)")
            except Exception as e:
                print(f"✗ Claude API configuration failed: {e}")
        else:
            print("⊗ Claude API key not provided - skipping")

        if openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=openai_api_key)
                self.active_platforms.append('chatgpt')
                print("✓ OpenAI API configured (with web search via Bing)")
            except Exception as e:
                print(f"✗ OpenAI API configuration failed: {e}")
        else:
            print("⊗ OpenAI API key not provided - skipping")

        if gemini_api_key:
            try:
                self.gemini_client = genai.Client(api_key=gemini_api_key)
                self.active_platforms.append('gemini')
                print("✓ Gemini API configured (with Google Search grounding)")
            except Exception as e:
                print(f"✗ Gemini API configuration failed: {e}")
        else:
            print("⊗ Gemini API key not provided - skipping")

        if perplexity_api_key:
            self.perplexity_api_key = perplexity_api_key
            self.active_platforms.append('perplexity')
            print("✓ Perplexity API configured (search-native with citations)")
        else:
            print("⊗ Perplexity API key not provided - skipping")

        if not self.active_platforms:
            print("\n⚠️  WARNING: No API keys configured! Please provide at least one API key.")

    def query_claude_with_search(self, prompt: str) -> Dict[str, Any]:
        """Query Claude with web search enabled."""
        if not self.claude_client:
            return None

        try:
            response = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search"
                }],
                messages=[{"role": "user", "content": prompt}]
            )

            # Extract response and sources
            answer = ""
            sources = []

            for block in response.content:
                if block.type == "text":
                    answer += block.text
                elif block.type == "tool_use" and block.name == "web_search":
                    # Claude used web search
                    sources.append({
                        "query": block.input.get("query", ""),
                        "type": "web_search"
                    })

            # Check if there's a follow-up response with search results
            if response.stop_reason == "tool_use":
                # Process tool results (simplified - in production you'd handle the full tool use cycle)
                sources.append({"note": "Web search was invoked"})

            return {
                "answer": answer,
                "sources": sources,
                "used_search": len(sources) > 0
            }

        except Exception as e:
            error_msg = str(e)
            if "credit balance is too low" in error_msg.lower() or "quota" in error_msg.lower():
                raise Exception("Insufficient credits/quota")
            raise Exception(error_msg[:200])

    def query_openai_with_search(self, prompt: str) -> Dict[str, Any]:
        """Query OpenAI with web search enabled (requires GPT-4 with browsing)."""
        if not self.openai_client:
            return None

        try:
            # Note: As of Jan 2025, OpenAI doesn't have direct web search in API
            # This would require function calling setup or using plugins
            # For now, we'll use the base model and note the limitation

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant. If you need current information, acknowledge that you cannot search the web via API but provide the best answer based on your training data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048
            )

            answer = response.choices[0].message.content

            return {
                "answer": answer,
                "sources": [],
                "used_search": False,
                "note": "OpenAI API doesn't support web search directly. Using base model knowledge."
            }

        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
                raise Exception("Insufficient quota/credits")
            raise Exception(error_msg[:200])

    def query_gemini_with_search(self, prompt: str) -> Dict[str, Any]:
        """Query Gemini with Google Search grounding."""
        if not self.gemini_client:
            return None

        try:
            # Use Google Search grounding
            response = self.gemini_client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt,
                config={
                    'tools': [{
                        'google_search': {}
                    }]
                }
            )

            answer = response.text
            sources = []

            # Extract grounding metadata if available
            if hasattr(response, 'grounding_metadata') and response.grounding_metadata:
                for chunk in response.grounding_metadata.grounding_chunks:
                    if hasattr(chunk, 'web'):
                        sources.append({
                            "url": chunk.web.uri if hasattr(chunk.web, 'uri') else "Unknown",
                            "title": chunk.web.title if hasattr(chunk.web, 'title') else "Unknown"
                        })

            return {
                "answer": answer,
                "sources": sources,
                "used_search": len(sources) > 0
            }

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                raise Exception("Rate limit/quota exceeded")
            raise Exception(error_msg[:200])

    def query_perplexity(self, prompt: str) -> Dict[str, Any]:
        """Query Perplexity API (search-native with built-in citations)."""
        if not self.perplexity_api_key:
            return None

        try:
            url = "https://api.perplexity.ai/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "llama-3.1-sonar-large-128k-online",  # Online model with search
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 2048,
                "search_domain_filter": [],
                "return_citations": True,
                "return_images": False
            }

            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()
            answer = data['choices'][0]['message']['content']

            # Extract citations
            sources = []
            if 'citations' in data:
                sources = data['citations']

            return {
                "answer": answer,
                "sources": sources,
                "used_search": True
            }

        except Exception as e:
            raise Exception(str(e)[:200])

    def query_all_platforms(self, prompt: str, use_search: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Send a query to all configured platforms with optional web search.
        Returns responses with sources.
        """
        results = {}
        errors = []

        print(f"\n{'🔍 Searching the web' if use_search else '💭 Using base knowledge'}...")

        # Query Claude
        if 'claude' in self.active_platforms:
            try:
                result = self.query_claude_with_search(prompt) if use_search else self.query_claude_base(prompt)
                results['claude'] = result
                search_status = "with search" if result.get('used_search') else "base knowledge"
                print(f"✓ Claude responded ({search_status})")
            except Exception as e:
                errors.append(f"Claude: {str(e)}")
                print(f"✗ Claude error: {str(e)}")

        # Query OpenAI
        if 'chatgpt' in self.active_platforms:
            try:
                result = self.query_openai_with_search(prompt)
                results['chatgpt'] = result
                print(f"✓ ChatGPT responded (note: no web search in API)")
            except Exception as e:
                errors.append(f"ChatGPT: {str(e)}")
                print(f"✗ ChatGPT error: {str(e)}")

        # Query Gemini
        if 'gemini' in self.active_platforms:
            try:
                result = self.query_gemini_with_search(prompt) if use_search else self.query_gemini_base(prompt)
                results['gemini'] = result
                search_status = "with Google Search" if result.get('used_search') else "base knowledge"
                print(f"✓ Gemini responded ({search_status})")
            except Exception as e:
                errors.append(f"Gemini: {str(e)}")
                print(f"✗ Gemini error: {str(e)}")

        # Query Perplexity
        if 'perplexity' in self.active_platforms:
            try:
                result = self.query_perplexity(prompt)
                results['perplexity'] = result
                print(f"✓ Perplexity responded (with citations)")
            except Exception as e:
                errors.append(f"Perplexity: {str(e)}")
                print(f"✗ Perplexity error: {str(e)}")

        if errors and not results:
            print("\n⚠️  All platforms failed:")
            for error in errors:
                print(f"  • {error}")

        return results

    def query_claude_base(self, prompt: str) -> Dict[str, Any]:
        """Query Claude without web search."""
        response = self.claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            "answer": response.content[0].text,
            "sources": [],
            "used_search": False
        }

    def query_gemini_base(self, prompt: str) -> Dict[str, Any]:
        """Query Gemini without search grounding."""
        response = self.gemini_client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt
        )
        return {
            "answer": response.text,
            "sources": [],
            "used_search": False
        }

    def display_results(self, results: Dict[str, Dict[str, Any]]):
        """Display individual responses with sources."""
        print("\n" + "=" * 60)
        print("INDIVIDUAL RESPONSES")
        print("=" * 60)

        for platform, result in results.items():
            print(f"\n{'─' * 60}")
            print(f"✓ {platform.upper()}")
            if result.get('used_search'):
                print(f"   🔍 Used web search")
            if result.get('note'):
                print(f"   ℹ️  {result['note']}")
            print(f"{'─' * 60}")

            print(result['answer'])

            # Display sources
            if result.get('sources') and len(result['sources']) > 0:
                print(f"\n📚 Sources ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'][:10], 1):  # Limit to 10
                    if isinstance(source, dict):
                        if 'url' in source:
                            print(f"  {i}. {source.get('title', 'Unknown')} - {source['url']}")
                        elif 'query' in source:
                            print(f"  {i}. Search query: {source['query']}")
                        else:
                            print(f"  {i}. {source}")
                    else:
                        print(f"  {i}. {source}")

    def synthesize_responses(self, results: Dict[str, Dict[str, Any]]) -> str:
        """
        Use Claude to synthesize all responses into a consensus answer.
        Includes all sources from all platforms.
        """
        if not results:
            return "❌ No successful responses received from any platform."

        if len(results) == 1:
            platform, result = list(results.items())[0]
            return f"✓ Only {platform.upper()} responded:\n\n{result['answer']}"

        # Collect all sources
        all_sources = []
        for platform, result in results.items():
            if result.get('sources'):
                for source in result['sources']:
                    if source not in all_sources:
                        all_sources.append({"platform": platform, "source": source})

        # Build synthesis prompt
        synthesis_prompt = "I asked multiple AI models the same question. Here are their responses:\n\n"

        for platform, result in results.items():
            synthesis_prompt += f"**{platform.upper()}** "
            if result.get('used_search'):
                synthesis_prompt += "(used web search)"
            synthesis_prompt += f":\n{result['answer']}\n\n"

        synthesis_prompt += "\nPlease synthesize these responses into a single, comprehensive answer that:\n"
        synthesis_prompt += "1. Identifies where the models agree and disagree\n"
        synthesis_prompt += "2. Evaluates which information seems most reliable (prioritize responses that used web search for current topics)\n"
        synthesis_prompt += "3. Provides the most accurate, up-to-date, and helpful consolidated response\n"
        synthesis_prompt += "4. Notes if some models had access to more current information than others\n"
        synthesis_prompt += "5. Is concise and well-organized"

        # Try to use Claude for synthesis
        if self.claude_client:
            try:
                response = self.claude_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2048,
                    messages=[{"role": "user", "content": synthesis_prompt}]
                )
                synthesis = response.content[0].text

                # Add aggregated sources
                if all_sources:
                    synthesis += "\n\n" + "=" * 60
                    synthesis += "\n📚 AGGREGATED SOURCES FROM ALL PLATFORMS"
                    synthesis += "\n" + "=" * 60 + "\n"
                    for i, item in enumerate(all_sources[:20], 1):  # Limit to 20
                        platform = item['platform']
                        source = item['source']
                        if isinstance(source, dict) and 'url' in source:
                            synthesis += f"\n{i}. [{platform}] {source.get('title', 'Unknown')}\n   {source['url']}"
                        elif isinstance(source, dict) and 'query' in source:
                            synthesis += f"\n{i}. [{platform}] Search: {source['query']}"
                        else:
                            synthesis += f"\n{i}. [{platform}] {source}"

                return "🔄 SYNTHESIZED CONSENSUS RESPONSE:\n\n" + synthesis

            except Exception as e:
                print(f"⚠️  Synthesis error: {str(e)[:100]}")

        # Fallback
        result = "🔄 COMBINED RESPONSES:\n\n"
        for platform, data in results.items():
            result += f"{'=' * 60}\n{platform.upper()}\n{'=' * 60}\n{data['answer']}\n\n"
        return result

    def save_results(self, query: str, results: Dict[str, Dict[str, Any]], synthesis: str,
                     filename: Optional[str] = None):
        """Save query results with sources to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multi_llm_search_{timestamp}.json"

        # Collect all sources
        all_sources = []
        for platform, result in results.items():
            if result.get('sources'):
                all_sources.extend([{"platform": platform, "source": s} for s in result['sources']])

        data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "platforms_used": list(results.keys()),
            "individual_responses": {
                platform: {
                    "answer": result['answer'],
                    "sources": result.get('sources', []),
                    "used_search": result.get('used_search', False)
                }
                for platform, result in results.items()
            },
            "all_sources": all_sources,
            "synthesized_response": synthesis
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {filename}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Multi-Platform AI Search & Consensus System")
    print("=" * 60)
    print("\n🔍 This version enables web search where available!")
    print("\n💡 Recommended API keys:")
    print("  • Claude: Best synthesis + web search (console.anthropic.com)")
    print("  • Perplexity: Search-native with citations (perplexity.ai)")
    print("  • Gemini: Google Search grounding (ai.google.dev)")
    print("  • OpenAI: Base knowledge only, no API search (platform.openai.com)")

    # Get API keys
    claude_key = os.getenv('ANTHROPIC_API_KEY') or input(
        "\nEnter Claude API key (recommended, or press Enter to skip): ").strip() or None
    openai_key = os.getenv('OPENAI_API_KEY') or input("Enter OpenAI API key (or press Enter to skip): ").strip() or None
    gemini_key = os.getenv('GEMINI_API_KEY') or input("Enter Gemini API key (or press Enter to skip): ").strip() or None
    perplexity_key = os.getenv('PERPLEXITY_API_KEY') or input(
        "Enter Perplexity API key (or press Enter to skip): ").strip() or None

    # Initialize extractor
    print("\n" + "=" * 60)
    extractor = ChatHistoryExtractor(
        claude_api_key=claude_key,
        openai_api_key=openai_key,
        gemini_api_key=gemini_key,
        perplexity_api_key=perplexity_key
    )

    if not extractor.active_platforms:
        print("\n❌ No platforms configured. Exiting...")
        return

    # Interactive query mode
    print("\n" + "=" * 60)
    print("Query Mode - Multi-Platform Search & Consensus")
    print("=" * 60)
    print(f"Active platforms: {', '.join(extractor.active_platforms)}")

    while True:
        query = input("\n🤔 Enter your question (or 'quit' to exit): ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not query:
            continue

        # Ask if user wants web search
        use_search = input("🔍 Enable web search? (y/n, default=y): ").strip().lower()
        use_search = use_search != 'n'

        results = extractor.query_all_platforms(query, use_search=use_search)

        if not results:
            print("\n❌ No successful responses. Please check your API keys and quotas.")
            continue

        # Display individual responses
        extractor.display_results(results)

        # Synthesize
        print("\n" + "=" * 60)
        print("SYNTHESIZING CONSENSUS...")
        print("=" * 60)

        synthesis = extractor.synthesize_responses(results)
        print("\n" + synthesis)

        # Save option
        save = input("\n💾 Save results? (y/n): ").strip().lower()
        if save == 'y':
            extractor.save_results(query, results, synthesis)


if __name__ == "__main__":
    main()