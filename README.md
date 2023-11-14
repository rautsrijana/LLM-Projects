# LLM-Projects
I have recently immersed myself in langchain agents, chains, and word embeddings to enhance my comprehension of creating language model-driven applications. 
This repo consist of Various LLM projects that i have been explored so far. 

## 3_Chat_with_multiple_pdf_using_langchain_and_llama2

In this blog post, we will explore how to build a chat functionality to query a PDF document using Langchain and Llama2.The goal is to create a chat interface where users can ask questions related to the PDF content, and the system will provide relevant answers based on the text in the PDF.

### Dataset

I have download the data from Venelin Valkov(https://www.youtube.com/@venelin_valkov) google drive. Those pdf file are extracted from there. 

PyPDF2: For reading PDF files

In the given notebook, In the first cell we see the CUDA VERSION where it shows 12.0 which help to set AutoGPTQ. AutoGPTQ is an easy-to-use LLMs quantization package with user-friendly apis, based on GPTQ algorithm.

For CUDA 12.1: pip install auto-gptq
For CUDA 11.8: pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/

### Langchain
Langchain is a Python library that provides various tools and functionalities for natural language processing (N.L.P.) tasks. It offers text-splitting capabilities, embedding generation, and integration with powerful N.L.P. models like OpenAI's GPT-3.5. F.A.I.S.S., on the other hand, is a library for efficient similarity search and clustering of dense vectors.


### Text Embeddings
Text embeddings are the heart and soul of Large Language Operations. Technically, we can work with language models with natural language but storing and retrieving natural language is highly inefficient.

For the Embedding I have used model called hkunlp/instructor-large(https://huggingface.co/hkunlp/instructor-large) from Massive Text Embedding Benchmark(MTEB) Leaderboard(https://huggingface.co/spaces/mteb/leaderboard). This model is in 14 th position
    
### Model - Llama 2 13B Chat - GPTQ(https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ)
Model creator: Meta Llama 2
Original model: Llama 2 13B Chat

In our case,
Branch - gptq-4bit-32g-actorder_True    
Bits - 4   
GS -  32   
aCT Order -  Yes    
Damp % - 0.01   
GPTQ Dataset -  wikitext  
Seq Len -   4096
Size - 8GB
ExLlama - Yes
Desc - 4-bit, with Act Order and group size 32g. Gives highest possible inference quality, with maximum VRAM usage.

This is our Final Pipeline
'''
text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15,
    streamer=streamer,
)
'''







