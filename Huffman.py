import heapq
import collections

class Node:
    def __init__(self, freq, char=None):
        self.freq = freq
        self.char = char
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(text):
    freq_dict = collections.Counter(text)
    heap = [Node(freq, char) for char, freq in freq_dict.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = Node(node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heap[0]

def build_huffman_codes(node, current_code="", huffman_codes={}):
    if node.char:
        huffman_codes[node.char] = current_code
        return huffman_codes

    build_huffman_codes(node.left, current_code + "0", huffman_codes)
    build_huffman_codes(node.right, current_code + "1", huffman_codes)
    return huffman_codes

def huffman_encode(text):
    if not text:
        return "", {}

    root = build_huffman_tree(text)
    huffman_codes = build_huffman_codes(root)
    encoded_text = "".join([huffman_codes[char] for char in text])
    return encoded_text, huffman_codes

def huffman_decode(encoded_text, huffman_codes):
    if not encoded_text:
        return ""

    reversed_codes = {v: k for k, v in huffman_codes.items()}
    decoded_text = ""
    current_code = ""

    for bit in encoded_text:
        current_code += bit
        if current_code in reversed_codes:
            decoded_text += reversed_codes[current_code]
            current_code = ""

    return decoded_text

# Example usage
text = "hello world"
encoded_text, huffman_codes = huffman_encode(text)
decoded_text = huffman_decode(encoded_text, huffman_codes)

print(f"Original text: {text}")
print(f"Encoded text: {encoded_text}")
print(f"Decoded text: {decoded_text}")
print(f"Huffman codes: {huffman_codes}")
