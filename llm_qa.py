from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

class LLMQA:
    def __init__(self, model_name='google/flan-t5-small'):
        print(f"Loading LLM model via LangChain: {model_name}")
        
        device = 0 if torch.cuda.is_available() else -1
        device_name = 'GPU' if device == 0 else 'CPU'
        
        try:
            print(f"Downloading/loading tokenizer and model (this may take a moment)...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,  # Reduce memory usage
                torch_dtype=torch.float32
            )
            
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                device=device,
                temperature=0.7
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
        
            self.prompt_template = """Based on the following context, answer the question. If the answer is not in the context, say "I cannot find this information in the document."

Context:
{context}

Question: {question}

Answer:"""
            
            print(f"LangChain LLM loaded on {device_name}")
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error loading model: {error_msg}")
            if "paging file" in error_msg.lower() or "memory" in error_msg.lower():
                print("\n⚠️  MEMORY ERROR DETECTED:")
                print("   - Your system may not have enough RAM to load this model")
                print("   - Try closing other applications")
                print("   - The app will fall back to SimpleQA mode")
            raise
    
    def generate_answer(self, query, context_chunks):
        context_text = "\n\n".join([
            f"[Source: {chunk['source']}]\n{chunk['content'][:500]}"
            for chunk in context_chunks[:3]
        ])
        
        prompt = self.prompt_template.format(
            context=context_text,
            question=query
        )
        
        try:
            result = self.llm.invoke(prompt)
            answer = result.strip()
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            answer = "Sorry, I encountered an error generating the answer."
        
        return answer
    
    def generate_answer_with_citations(self, query, search_results):
        
        context_chunks = [result['chunk'] for result in search_results]
        
        answer = self.generate_answer(query, context_chunks)
        
        # Format the answer nicely
        formatted_answer = f"💬 **Answer:**\n\n{answer}"
       
        citations = []
        for i, result in enumerate(search_results[:3]):
            chunk = result['chunk']
            content = chunk.get('content', '')
            citations.append({
                'rank': i + 1,
                'source': chunk['source'],
                'page': chunk.get('page', 'N/A'),
                'type': chunk.get('type', 'text'),
                'relevance_score': result['score'],
                'preview': content[:100].strip() + "..." if len(content) > 100 else content.strip()
            })
        
        return {
            'answer': formatted_answer,
            'citations': citations,
            'context_used': len(context_chunks)
        }

class SimpleQA:
    def __init__(self):
        print("✓ SimpleQA initialized (no LLM model required)")
    
    def generate_answer_with_citations(self, query, search_results):
        if not search_results:
            return {
                'answer': "❌ **No relevant information found in the document.**",
                'citations': [],
                'context_used': 0
            }
        top_chunks = search_results[:3]
        
        # Build formatted answer with excerpts
        answer_parts = []
        answer_parts.append("📄 **Here's what I found in the document:**\n")
        
        for i, result in enumerate(top_chunks):
            chunk = result['chunk']
            content = chunk['content'].strip()
            
            # Clean up the content - remove excessive whitespace
            content = ' '.join(content.split())
            
            # Get a meaningful snippet (up to 300 chars)
            snippet = content[:300].strip()
            if len(content) > 300:
                # Try to cut at a sentence or word boundary
                last_period = snippet.rfind('.')
                last_space = snippet.rfind(' ')
                if last_period > 200:
                    snippet = snippet[:last_period + 1]
                elif last_space > 250:
                    snippet = snippet[:last_space] + "..."
                else:
                    snippet = snippet + "..."
            
            # Format each excerpt as a quote block
            source_type = chunk.get('type', 'text').capitalize()
            type_emoji = "📝" if source_type == "Text" else "📊" if source_type == "Table" else "🖼️"
            
            answer_parts.append(
                f"**{i+1}. {type_emoji} {chunk['source']}**\n"
                f"> {snippet}\n"
            )
        
        answer = "\n".join(answer_parts)
        
        # Build citations with more details
        citations = []
        for i, result in enumerate(top_chunks):
            chunk = result['chunk']
            citations.append({
                'rank': i + 1,
                'source': chunk['source'],
                'page': chunk.get('page', 'N/A'),
                'type': chunk.get('type', 'text'),
                'relevance_score': result['score'],
                'preview': chunk['content'][:100].strip() + "..." if len(chunk['content']) > 100 else chunk['content'].strip()
            })
        
        return {
            'answer': answer,
            'citations': citations,
            'context_used': len(search_results)
        }

if __name__ == "__main__":

    test_results = [
        {
            'chunk': {
                'content': 'Qatar economy grew by 5% in 2024 driven by strong non-hydrocarbon sector growth.',
                'page': 1,
                'type': 'text',
                'source': 'Page 1'
            },
            'score': 0.85
        },
        {
            'chunk': {
                'content': 'The banking sector remains healthy with strong capital ratios.',
                'page': 2,
                'type': 'text',
                'source': 'Page 2'
            },
            'score': 0.72
        }
    ]
    try:
        print("\n1. Test ")
        qa = LLMQA()
        result = qa.generate_answer_with_citations("What is Qatar's growth?", test_results)
        print(f"\nAnswer: {result['answer']}")
        print(f"Citations: {len(result['citations'])} sources")
        
    except Exception as e:
        print(f"\nLangChain LLMQA failed: {e}")
        print("\n2. Test Fallback ")
        qa = SimpleQA()
        result = qa.generate_answer_with_citations("What is Qatar's growth?", test_results)
        print(f"\nAnswer: {result['answer']}")
        print(f"Citations: {len(result['citations'])} sources")