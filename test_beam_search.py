import unittest
from typing import List, Dict
from beam_search import (
    format_conversation_history,
    format_candidates,
    generate_single_candidate,
    generate_candidates,
    evaluate_candidates,
    requires_python_execution,
    get_python_code,
    process_beam_search_step
)
from e2b_code_interpreter import Sandbox
import os

class TestBeamSearch(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = "openai/gpt-4o-mini"
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.sandbox = Sandbox()
        
        # Sample task and conversation history
        self.task = "Write a Python function to calculate the factorial of a number."
        self.conversation_history = [
            {
                'role': 'assistant',
                'content': "Let's break this down step by step."
            }
        ]
        
        # Sample tools
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "python",
                    "description": "Execute Python code",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"}
                        },
                        "required": ["code"]
                    }
                }
            }
        ]

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'sandbox'):
            self.sandbox.close()

    def test_format_conversation_history(self):
        """Test conversation history formatting."""
        history = [
            {'role': 'system', 'content': 'System message'},
            {'role': 'assistant', 'content': 'Assistant message'},
            {'role': 'user', 'content': 'User message'}
        ]
        formatted = format_conversation_history(history)
        self.assertIn('ASSISTANT: Assistant message', formatted)
        self.assertIn('USER: User message', formatted)
        self.assertNotIn('SYSTEM:', formatted)

    def test_format_candidates(self):
        """Test candidate formatting."""
        candidates = [
            {
                'content': 'First thought',
                'tool_calls': [{
                    'function': {
                        'name': 'python',
                        'arguments': '{"code": "print(1)"}'
                    }
                }]
            }
        ]
        formatted = format_candidates(candidates)
        self.assertIn('CANDIDATE 0:', formatted)
        self.assertIn('First thought', formatted)
        self.assertIn('Tool Calls:', formatted)

    def test_generate_candidates(self):
        """Test concurrent candidate generation."""
        num_candidates = 3
        candidates = generate_candidates(
            task=self.task,
            conversation_history=self.conversation_history,
            num_candidates=num_candidates,
            api_key=self.api_key,
            model=self.model,
            api_url=self.api_url,
            tools=self.tools
        )
        
        self.assertEqual(len(candidates), num_candidates)
        for candidate in candidates:
            self.assertIn('role', candidate)
            self.assertIn('content', candidate)

    def test_evaluate_candidates(self):
        """Test candidate evaluation."""
        candidates = [
            {'content': 'First approach'},
            {'content': 'Second approach'}
        ]
        
        best_index, explanation = evaluate_candidates(
            candidates=candidates,
            task=self.task,
            conversation_history=self.conversation_history,
            api_key=self.api_key,
            model=self.model,
            api_url=self.api_url
        )
        
        self.assertIsInstance(best_index, int)
        self.assertIsInstance(explanation, str)
        self.assertLess(best_index, len(candidates))

    def test_requires_python_execution(self):
        """Test Python execution detection."""
        candidate_with_python = {
            'tool_calls': [{
                'function': {
                    'name': 'python',
                    'arguments': '{"code": "print(1)"}'
                }
            }]
        }
        candidate_without_python = {
            'tool_calls': [{
                'function': {
                    'name': 'other_tool',
                    'arguments': '{}'
                }
            }]
        }
        
        self.assertTrue(requires_python_execution(candidate_with_python))
        self.assertFalse(requires_python_execution(candidate_without_python))

    def test_get_python_code(self):
        """Test Python code extraction."""
        candidate = {
            'tool_calls': [{
                'function': {
                    'name': 'python',
                    'arguments': '{"code": "print(1)"}'
                }
            }]
        }
        code = get_python_code(candidate)
        self.assertEqual(code, "print(1)")

    def test_process_beam_search_step(self):
        """Test full beam search step processing."""
        best_candidate, new_branch = process_beam_search_step(
            task=self.task,
            conversation_history=self.conversation_history,
            num_candidates=3,
            current_branch='root',
            api_key=self.api_key,
            model=self.model,
            api_url=self.api_url,
            tools=self.tools,
            sandbox=self.sandbox,
            verbose=True
        )
        
        self.assertIn('role', best_candidate)
        self.assertIn('content', best_candidate)
        self.assertIsInstance(new_branch, str)

if __name__ == '__main__':
    unittest.main() 