#!/usr/bin/env python3
"""
Integration test to catch API mismatches between components.

This test verifies that main.py only calls methods that actually exist
on the component classes, preventing runtime AttributeError issues.
"""

import sys
import ast
import inspect
from typing import Dict, Set, List

sys.path.append('.')

def extract_method_calls(file_path: str, target_attr: str) -> Set[str]:
    """Extract method calls on a specific attribute from Python source code."""
    with open(file_path, 'r') as f:
        source = f.read()
    
    tree = ast.parse(source)
    method_calls = set()
    
    class MethodCallVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            if (isinstance(node.func, ast.Attribute) and 
                isinstance(node.func.value, ast.Attribute) and
                isinstance(node.func.value.value, ast.Name) and
                node.func.value.value.id == 'self' and
                node.func.value.attr == target_attr):
                method_calls.add(node.func.attr)
            self.generic_visit(node)
    
    visitor = MethodCallVisitor()
    visitor.visit(tree)
    return method_calls

def get_class_methods(module_path: str, class_name: str) -> Set[str]:
    """Get public method names from a class, handling import failures gracefully."""
    try:
        # Try to import and inspect the class
        spec = importlib.util.spec_from_file_location("module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        cls = getattr(module, class_name)
        return {name for name in dir(cls) if not name.startswith('_') and callable(getattr(cls, name))}
    
    except Exception:
        # If import fails, parse the class definition from source
        with open(module_path, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        methods = set()
        
        class ClassVisitor(ast.NodeVisitor):
            def visit_ClassDef(self, node):
                if node.name == class_name:
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith('_'):
                            methods.add(item.name)
                self.generic_visit(node)
        
        visitor = ClassVisitor()
        visitor.visit(tree)
        return methods

def test_api_compatibility():
    """Test API compatibility between main.py and its components."""
    
    print("🔍 Testing API compatibility...")
    
    # Test cases: (caller_file, target_attr, target_module, target_class)
    test_cases = [
        ('src/main.py', 'ui', 'src/ui/terminal.py', 'TerminalUI'),
        ('src/main.py', 'recorder', 'src/audio/recorder.py', 'AudioRecorder'), 
        ('src/main.py', 'transcriber', 'src/audio/transcriber.py', 'WhisperTranscriber'),
        ('src/main.py', 'cleaner', 'src/cleanup/cleaner.py', 'TextCleaner'),
    ]
    
    all_passed = True
    
    for caller_file, attr_name, target_module, target_class in test_cases:
        print(f"\n📋 Testing {caller_file} -> {attr_name} ({target_class})")
        
        try:
            # Get method calls from main.py
            called_methods = extract_method_calls(caller_file, attr_name)
            print(f"  📞 Called methods: {called_methods}")
            
            # Get available methods from target class
            available_methods = get_class_methods(target_module, target_class)
            print(f"  ✅ Available methods: {available_methods}")
            
            # Check for mismatches
            missing_methods = called_methods - available_methods
            if missing_methods:
                print(f"  ❌ MISSING METHODS: {missing_methods}")
                all_passed = False
            else:
                print(f"  ✅ All method calls valid")
                
        except Exception as e:
            print(f"  ❌ Error testing {caller_file} -> {attr_name}: {e}")
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    # Add importlib for dynamic imports
    import importlib.util
    
    if test_api_compatibility():
        print("\n🎉 All API compatibility tests passed!")
        sys.exit(0)
    else:
        print("\n💥 API compatibility tests failed!")
        sys.exit(1)