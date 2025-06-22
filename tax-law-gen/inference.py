#!/usr/bin/env python3
"""
Phi-2 Tax Law Q&A Inference Script
Optimized for M4 MacBook Pro with Metal Performance Shaders
"""

import os
import json
import torch
import logging
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TaxLawQAInference:
    """Tax Law Q&A Inference Class for Phi-2"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the inference model
        
        Args:
            model_path: Path to the fine-tuned model
            device: Device to use ('auto', 'mps', 'cpu')
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None
        self.prompt_template = "### 질문: {question}\n\n### 답변:"
        
        logger.info(f"Initializing model from {model_path}")
        logger.info(f"Using device: {self.device}")
        
        self._load_model()
    
    def _setup_device(self, device: str) -> str:
        """Setup the best available device"""
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            logger.info("Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/phi-2",
                torch_dtype=torch.float16,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Load LoRA weights
            logger.info("Loading LoRA weights...")
            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_path,
                torch_dtype=torch.float16
            )
            
            # Move to device if not using device_map
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_answer(
        self, 
        question: str, 
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True
    ) -> str:
        """
        Generate answer for a given question
        
        Args:
            question: The question to answer
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty
            do_sample: Whether to use sampling
            
        Returns:
            Generated answer
        """
        try:
            # Format prompt
            prompt = self.prompt_template.format(question=question)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # Move to device
            if self.device != "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                start_time = time.time()
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                generation_time = time.time() - start_time
            
            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            logger.info(f"Generation completed in {generation_time:.2f}s")
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"오류가 발생했습니다: {str(e)}"
    
    def batch_generate(
        self, 
        questions: List[str],
        **kwargs
    ) -> List[str]:
        """
        Generate answers for multiple questions
        
        Args:
            questions: List of questions
            **kwargs: Generation parameters
            
        Returns:
            List of generated answers
        """
        answers = []
        total_time = 0
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            start_time = time.time()
            
            answer = self.generate_answer(question, **kwargs)
            answers.append(answer)
            
            question_time = time.time() - start_time
            total_time += question_time
            
            logger.info(f"Question {i+1} completed in {question_time:.2f}s")
        
        logger.info(f"Batch processing completed in {total_time:.2f}s")
        return answers
    
    def interactive_mode(self):
        """Interactive mode for testing"""
        print("\n=== 세법 Q&A 시스템 ===")
        print("질문을 입력하세요. 'quit' 또는 'exit'를 입력하면 종료됩니다.\n")
        
        while True:
            try:
                question = input("질문: ").strip()
                
                if question.lower() in ['quit', 'exit', '종료']:
                    print("시스템을 종료합니다.")
                    break
                
                if not question:
                    continue
                
                print("답변 생성 중...")
                answer = self.generate_answer(question)
                print(f"\n답변: {answer}\n")
                
            except KeyboardInterrupt:
                print("\n시스템을 종료합니다.")
                break
            except Exception as e:
                print(f"오류가 발생했습니다: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phi-2 Tax Law Q&A Inference")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="./phi2-tax-law-finetuned",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "mps", "cpu"],
        help="Device to use for inference"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--question", 
        type=str,
        help="Single question to answer"
    )
    parser.add_argument(
        "--questions_file", 
        type=str,
        help="File containing questions (one per line)"
    )
    parser.add_argument(
        "--output_file", 
        type=str,
        help="Output file for answers"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=512,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Sampling temperature"
    )
    
    args = parser.parse_args()
    
    # Initialize inference
    try:
        inference = TaxLawQAInference(args.model_path, args.device)
    except Exception as e:
        logger.error(f"Failed to initialize inference: {e}")
        return
    
    # Run based on mode
    if args.interactive:
        inference.interactive_mode()
    
    elif args.question:
        answer = inference.generate_answer(
            args.question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        print(f"\n질문: {args.question}")
        print(f"답변: {answer}")
    
    elif args.questions_file:
        if not os.path.exists(args.questions_file):
            logger.error(f"Questions file not found: {args.questions_file}")
            return
        
        with open(args.questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        answers = inference.batch_generate(
            questions,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        
        # Save results
        results = []
        for q, a in zip(questions, answers):
            results.append({
                "question": q,
                "answer": a
            })
        
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {args.output_file}")
        else:
            for i, (q, a) in enumerate(zip(questions, answers)):
                print(f"\n질문 {i+1}: {q}")
                print(f"답변 {i+1}: {a}")
    
    else:
        print("사용법:")
        print("  --interactive: 대화형 모드")
        print("  --question '질문': 단일 질문 답변")
        print("  --questions_file 파일명: 파일의 질문들 답변")

if __name__ == "__main__":
    main() 