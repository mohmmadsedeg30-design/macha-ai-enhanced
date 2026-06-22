import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from macha.model.transformer import MachaModel, MachaTokenizer
from macha.utils.responses import SmartResponseSystem

def test_tokenizer():
    t = MachaTokenizer(vocab_size=100)
    t.fit(["hello", "world", "test"])
    assert len(t.encode("hello")) > 0
    print("Tokenizer test passed")

def test_model():
    m = MachaModel(vocab_size=50, d_model=32, num_heads=2, num_layers=1)
    assert sum(p.numel() for p in m.parameters()) > 0
    print("Model test passed")

def test_responses():
    s = SmartResponseSystem()
    r = s.get_response("hello")
    assert len(r) > 0
    print("Response system test passed")

if __name__ == "__main__":
    test_tokenizer()
    test_model()
    test_responses()
    print("All tests passed!")
