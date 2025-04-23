import logging
import os
import json
import spacy
from datetime import datetime
from .base_tool import Tool, ToolExecutionError

logger = logging.getLogger("gemini_agent")

class NaturalLanguageProcessingTool(Tool):
    def __init__(self):
        super().__init__(
            name="process_text",
            description="Perform natural language processing tasks like sentiment analysis, entity recognition, and summarization using spaCy",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to be analyzed"
                    },
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of NLP analysis to perform",
                        "enum": ["entities", "pos", "dependency", "summarization", "similarity"]
                    },
                    "compare_text": {
                        "type": "string",
                        "description": "Secondary text for similarity comparison (required if analysis_type is 'similarity')"
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Format for the output (json or text)",
                        "enum": ["json", "text"],
                        "default": "json"
                    }
                },
                "required": ["text", "analysis_type"]
            }
        )
        # Load spaCy model once during initialization
        try:
            self.nlp = spacy.load("en_core_web_md")  # Medium-sized model with word vectors
            logger.info("spaCy model loaded successfully")
        except OSError:
            # If model not found, try to download it
            logger.warning("spaCy model not found, attempting to download...")
            try:
                import subprocess
                subprocess.check_call(["python", "-m", "spacy", "download", "en_core_web_md"])
                self.nlp = spacy.load("en_core_web_md")
                logger.info("spaCy model downloaded and loaded successfully")
            except Exception as e:
                logger.error(f"Failed to download spaCy model: {e}")
                raise ToolExecutionError("Failed to initialize NLP tool: Could not load or download spaCy model")
    
    def execute(self, **kwargs):
        self.validate_args(kwargs)
        
        try:
            text = kwargs.get("text")
            analysis_type = kwargs.get("analysis_type")
            output_format = kwargs.get("output_format", "json")
            
            logger.info(f"Performing {analysis_type} analysis on text")
            
            # Process the text with spaCy
            doc = self.nlp(text)
            
            result = {}
            
            # Perform analysis based on type
            if analysis_type == "entities":
                entities = [{"text": ent.text, "start": ent.start_char, "end": ent.end_char, 
                            "type": ent.label_, "description": spacy.explain(ent.label_)}
                           for ent in doc.ents]
                result = {"entities": entities}
                
            elif analysis_type == "pos":
                tokens = [{"text": token.text, "pos": token.pos_, 
                          "tag": token.tag_, "description": spacy.explain(token.tag_)}
                         for token in doc]
                result = {"tokens": tokens}
                
            elif analysis_type == "dependency":
                dependencies = [{"text": token.text, "dep": token.dep_,
                               "head": token.head.text, "description": spacy.explain(token.dep_)}
                              for token in doc]
                result = {"dependencies": dependencies}
                
            elif analysis_type == "summarization":
                # Simple extractive summarization by selecting sentences with highest scores
                # Based on TF-IDF-like approach
                from collections import Counter
                word_freq = Counter([token.text.lower() for token in doc if not token.is_stop and token.is_alpha])
                sent_scores = {}
                for sent in doc.sents:
                    for word in sent:
                        if word.text.lower() in word_freq:
                            if sent in sent_scores:
                                sent_scores[sent] += word_freq[word.text.lower()]
                            else:
                                sent_scores[sent] = word_freq[word.text.lower()]
                
                # Get top 3 sentences or fewer if text is short
                summary_sentences = sorted(sent_scores, key=sent_scores.get, reverse=True)[:min(3, len(sent_scores))]
                summary = " ".join([sent.text for sent in sorted(summary_sentences, key=lambda x: x.start)])
                result = {"summary": summary}
                
            elif analysis_type == "similarity":
                compare_text = kwargs.get("compare_text")
                if not compare_text:
                    raise ToolExecutionError("compare_text is required for similarity analysis")
                
                compare_doc = self.nlp(compare_text)
                similarity_score = doc.similarity(compare_doc)
                result = {"similarity_score": similarity_score}
            
            # Store in vector DB if available
            try:
                from __main__ import vector_db
                if vector_db.is_ready():
                    vector_db.add(
                        f"NLP {analysis_type} analysis result",
                        {
                            "type": "nlp_analysis",
                            "analysis_type": analysis_type,
                            "input_text": text[:100] + "..." if len(text) > 100 else text,
                            "result_summary": str(result)[:100] + "..." if len(str(result)) > 100 else str(result),
                            "time": datetime.now().isoformat()
                        }
                    )
            except ImportError:
                pass
            
            # Format response based on output format
            if output_format == "text":
                if analysis_type == "entities":
                    entities_text = ", ".join([f"{e['text']} ({e['type']})" for e in result['entities']])
                    return f"Entities found: {entities_text}"
                elif analysis_type == "pos":
                    return "\n".join([f"{t['text']}: {t['pos']} ({t['tag']} - {t['description']})" for t in result['tokens']])
                elif analysis_type == "dependency":
                    return "\n".join([f"{d['text']} --{d['dep']}--> {d['head']}" for d in result['dependencies']])
                elif analysis_type == "summarization":
                    return f"Summary: {result['summary']}"
                elif analysis_type == "similarity":
                    return f"Similarity score: {result['similarity_score']:.4f}"
            else:
                return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"NLP processing error: {e}", exc_info=True)
            raise ToolExecutionError(f"Failed to process text: {e}")