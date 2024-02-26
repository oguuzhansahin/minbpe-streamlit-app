import streamlit as st
from basic import BasicTokenizer
from regex_ import RegexTokenizer

def display_tokens(tokenizer, encoded_tokens: list) -> None:
    # Bu fonksiyonun kullanımını aşağıdaki örnekte gösterdim.
    for token in encoded_tokens:
        st.markdown(f"<span style='color: red;'>{tokenizer.decode([token])}</span>", unsafe_allow_html=True)

def main():
    st.markdown("<h1 style='text-align: center;'>Training BPE Tokenizer with Your Data</h1>", unsafe_allow_html=True)
        
    vocab_size = st.sidebar.number_input("Vocab Size", min_value=257, max_value=30000, value=512)
    tokenizer_type = st.sidebar.radio("Tokenizer Type", ('basic', 'regex'), index=0)
    
    # Tokenizer initialization based on selection
    tokenizer = BasicTokenizer() if tokenizer_type == "basic" else RegexTokenizer()
    
    st.subheader("Upload txt")
    uploaded_files = st.file_uploader("Upload file/s", accept_multiple_files=True, type=['txt'])
    
    all_texts = [file.getvalue().decode("utf-8") for file in uploaded_files if file is not None]
    all_texts = " ".join(all_texts)
    if all_texts:  # Displaying text preview only if there's any text
        st.text_area("First 1000 chars of the given text: ", all_texts[:1000], height=150, disabled=True)
    
    if st.button("Train", help="Train the tokenizer with uploaded texts"):
        
        tokenizer.train(vocab_size=vocab_size, text=all_texts)
        tokenizer.save(file_prefix="test_tokenizer")
        st.success("Tokenizer trained and saved successfully!")
             
    test_text = st.text_area("Enter test text for tokenization:", height=150)  # Increased height for better input space
    
    if st.button("Tokenize", help="Tokenize the entered test text"):
        tokenizer.load(model_file="test_tokenizer.model")

        encoded_tokens = tokenizer.encode(test_text)
        # Displaying encoded tokens with styled HTML
        colored_text = ''.join(f'<span style="color: #{hash(tokenizer.decode([token]))%0xFFFFFF:06x}; font-size: 24px; font-weight: bold; font-family: Arial, sans-serif;">{tokenizer.decode([token])}</span> ' for token in encoded_tokens)
        st.markdown(colored_text, unsafe_allow_html=True)
                
if __name__ == "__main__":
    main()
