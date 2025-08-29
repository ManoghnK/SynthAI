# SynthAI
SynthAI: Multi-modal AI for personalized learning. Generates lessons, images, quizzes, &amp; audio on any topic using fine-tuned LLMs &amp; Stable Diffusion and RAG.
SynthAI: Educational Content Synthesizer
🎯 Objective
The primary objective of SynthAI is to revolutionize educational content creation by providing an automated, intelligent system capable of generating comprehensive and engaging learning materials. This project aims to synthesize detailed lessons, relevant images, interactive quizzes, and audio narrations for any given topic, making learning more accessible and personalized. The core idea is to leverage advanced Generative AI models to create a rich, multi-modal educational experience.
💡 Methodology
SynthAI employs a multi-faceted approach, integrating various Generative AI models and techniques:
1. Dataset Creation for Fine-tuning
To ensure the lesson generation model is proficient in producing high-quality educational content, a specialized dataset was created.
Data Collection: A diverse list of 60 educational topics was curated across four domains: Science, Mathematics, Computer Science, and Humanities. Examples include "Photosynthesis," "Pythagorean Theorem," "Data Structures Overview," and "Renaissance Art Period."
Prompt Engineering: For each topic, a detailed prompt was crafted using the Llama-3.1-8B-Instruct model via the Hugging Face Inference Client. This prompt instructed the model to act as an "expert educational content creator" and generate a comprehensive lesson (500-800 words) formatted in Markdown, including:
A descriptive level-1 heading.
An introduction explaining relevance.
3-5 key concepts with clear explanations.
Real-world applications or examples.
Practical examples, formulas, or code snippets (if applicable).
A concise summary.
Data Generation & Validation: Lessons were generated in chunks of 20 topics with exponential backoff for API stability. Each generated lesson underwent basic validation (e.g., minimum word count) and was saved. The final dataset was then validated for domain distribution, lesson word count statistics (average, min, max lengths), and Markdown formatting (headers, lists).
Hugging Face Dataset Conversion: The validated JSON dataset was converted into a Hugging Face Dataset object and split into training (70%), validation (15%), and test (15%) sets.
2. Fine-tuning the Lesson Generation Model
A base Large Language Model (LLM) was fine-tuned to generate high-quality educational lessons.
Base Model Selection: TinyLlama/TinyLlama-1.1B-Chat-v1.0 was chosen as the base model due to its efficiency and suitability for fine-tuning on consumer-grade GPUs.
Quantization: To optimize memory usage and speed up training, the model was loaded with 8-bit quantization using BitsAndBytesConfig. A fallback to float16 was implemented if 8-bit loading failed.
LoRA (Low-Rank Adaptation): Instead of full fine-tuning, LoRA was employed. This technique significantly reduces the number of trainable parameters by injecting small, trainable matrices into the transformer layers, making fine-tuning more efficient. The LoraConfig specified:
r=8 (LoRA rank)
lora_alpha=32 (scaling factor)
target_modules=["q_proj", "v_proj"] (applying LoRA to query and value projections)
lora_dropout=0.05
bias="none"
task_type=TaskType.CAUSAL_LM
Training Configuration: The TrainingArguments were set for a minimal yet effective training run, including:
output_dir: For saving model checkpoints and logs.
per_device_train_batch_size=1, per_device_eval_batch_size=1: Due to GPU memory constraints.
learning_rate=2e-4.
num_train_epochs=1.
logging_steps=10.
fp16=True: For faster training with mixed precision.
report_to="none": To disable external reporting.
Training Process: The Trainer from the transformers library was used to manage the training loop, including data collation (DataCollatorForLanguageModeling). The model was trained for one epoch, and the fine-tuned LoRA adapters and tokenizer were saved to Google Drive and pushed to the Hugging Face Hub under the repository Manoghn/tinyllama-lesson-synthesizer.
3. Multi-modal Content Generation
Once the lesson generation model is fine-tuned, SynthAI orchestrates the creation of a complete educational package.
a. Lesson Generation
The fine-tuned TinyLlama model (Manoghn/tinyllama-lesson-synthesizer) is used to generate detailed, Markdown-formatted lessons based on a user-provided topic. The prompt structure ensures the output adheres to educational standards.
b. Image Generation
Dynamic Image Suggestions: Instead of static prompts, the fine-tuned TinyLlama model dynamically analyzes the generated lesson text to suggest relevant image concepts. The AI is prompted to act as an "educational content expert" and provide descriptive titles and detailed visual descriptions for 3 images.
Context-Aware Styling: The image generation prompts are enhanced with context-aware artistic styles (e.g., "historical painting" for history topics, "scientific illustration" for science topics) and quality modifiers ("masterpiece, best quality, ultra-detailed").
Model: StableDiffusionPipeline (stabilityai/stable-diffusion-2-1) is used for image generation.
Parameters: Improved parameters are used for higher quality: num_inference_steps=50, guidance_scale=8.0, height=768, width=768.
Negative Prompts: An enhanced negative prompt is included to avoid common generation artifacts (e.g., "text overlay, watermark, low quality, blurry, distorted").
Post-processing: Generated images undergo minor post-processing (sharpness and contrast enhancement) using PIL for better clarity.
Fallback: If AI suggestions are insufficient, a fallback mechanism extracts key terms from the lesson text to generate generic but relevant image prompts.
c. Quiz Generation with RAG
Quiz generation is a critical component, utilizing a specialized model and a Retrieval-Augmented Generation (RAG) approach for accuracy.
Specialized Quiz Model: A dedicated model is initialized for quiz generation. The system attempts to load models in order of preference:
mistralai/Mistral-7B-Instruct-v0.1
meta-llama/Llama-2-7b-chat-hf
HuggingFaceH4/zephyr-7b-beta
TheBloke/Llama-2-7B-Chat-GPTQ (quantized version)
google/flan-t5-xl
google/flan-t5-large
This ensures a robust quiz generation capability, with 8-bit quantization applied where appropriate.
Retrieval-Augmented Generation (RAG):
Vector Store Creation: The generated lesson text is chunked using RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=50). These chunks are then embedded using HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2) and stored in a FAISS vector store.
Contextual Questioning: When generating quiz questions, relevant chunks are retrieved from this vector store based on the question type and topic. This ensures that the generated questions are directly grounded in the lesson content.
Diverse Question Types: The system generates a distribution of quiz questions, including:
Multiple Choice (30%): Questions with four options and a correct answer.
True/False (25%): Statements requiring a True/False response.
Fill-in-the-Blank (25%): Sentences with a missing word/phrase.
Short Answer (20%): Questions requiring brief, descriptive answers.
Prompt Structure: Type-specific prompts are used to guide the quiz model in generating questions in the desired format.
Parsing and Fallback: The model's raw output is parsed to extract structured questions and answers. A robust fallback mechanism is in place to generate questions (e.g., True/False, fill-in-the-blank, simple Q&A) from the lesson text if the specialized quiz model fails or doesn't produce enough questions. This fallback leverages sentence extraction, key term identification (bold terms, capitalized phrases), and definition extraction.
d. Audio Narration
gTTS Integration: The gTTS (Google Text-to-Speech) library is used to generate audio narration for the lesson.
Text Cleaning: The lesson text is cleaned (removing Markdown formatting, replacing newlines with periods) to ensure natural-sounding speech.
e. Comprehensive Study Guide
A Markdown-formatted study guide is compiled, integrating:
A summary of the generated lesson.
All generated quiz questions, grouped by type, with collapsible answer sections.
Question statistics (total, types, percentages).
Detailed study tips for each question type and general study strategies (active recall, spaced repetition, etc.).
Dynamic learning objectives extracted from key concepts in the quiz questions and lesson text.
Suggestions for additional resources.
A dedicated notes section for learners.
A self-assessment section for tracking understanding.
💻 Project Structure
The project primarily consists of Python code within Jupyter notebooks, which are then used to generate standalone Python files for deployment.
SynthAI_prefinal.ipynb: The main development notebook containing the code for model initialization, lesson generation, image suggestion, quiz generation, audio generation, and study guide creation. This notebook also generates Another_copy_of_text_generator_Dataset_creation.py (backend logic), study_generator.py (wrapper), and app.py (Streamlit frontend).
Testing_model.ipynb: This notebook focuses on the dataset creation and fine-tuning process, including:
Package installation and Hugging Face token setup.
Definition of educational topics across domains.
Generation of the lesson_dataset_complete.json file using Llama-3.1.
Dataset validation and conversion to Hugging Face Dataset format.
Fine-tuning of TinyLlama with LoRA.
Saving the fine-tuned model to Google Drive and pushing to Hugging Face Hub.
Testing the fine-tuned model for lesson generation.
Generating an HTML summary of generated lessons.
Another_copy_of_text_generator_Dataset_creation.py: Contains the core backend logic for all content generation functionalities (lesson, image, quiz, audio, study guide).
study_generator.py: A simple wrapper to import generate_study_guide from the backend module.
app.py: The Streamlit application that provides a user interface for entering a topic and displaying the generated multi-modal study guide. It includes custom CSS for a modern, responsive design and uses 21st.dev icons.
hftoken.txt: A file to securely store the Hugging Face token.
lesson_dataset_complete.json: The generated dataset used for fine-tuning.
🚀 Deployment
The Streamlit application (app.py) can be deployed using ngrok or localtunnel for public access, as demonstrated in the SynthAI_prefinal.ipynb notebook. The deployment steps involve:
Installing pyngrok or localtunnel.
Authenticating with an ngrok authtoken.
Running the Streamlit app in the background.
Exposing the Streamlit app via a public URL.
⚠️ Limitations & Future Considerations
This project was developed using the Colab free tier with a T4 GPU, which presented several computational and deployment challenges:
RAM Limitations: Running multiple large models (TinyLlama, Stable Diffusion, and a specialized quiz model) simultaneously often led to Out-Of-Memory (OOM) errors or severe performance degradation due to insufficient RAM. This necessitated careful memory management, including CPU offloading for Stable Diffusion and 8-bit quantization for LLMs.
Future Handling: With more resources, a dedicated GPU instance with higher VRAM (e.g., A100, H100) would allow for larger batch sizes, higher-resolution image generation, and the simultaneous loading of more powerful LLMs, significantly improving performance and reducing OOM issues.
Local Hosting Challenges (Mac M4 Chip): Hosting the application locally on a Mac M4 chip, while powerful for certain tasks, was a hassle for public access and consistent performance, especially when dealing with the heavy computational load of multiple AI models.
Future Handling: For robust public hosting, deploying the Streamlit application on a cloud platform like Google Cloud Run, AWS App Runner, or Hugging Face Spaces would provide scalable, managed infrastructure, eliminating local hosting complexities and ensuring better uptime and performance.
ngrok Performance Overhead: Using ngrok for exposing the local Streamlit application to the public internet introduced significant latency and slowed down the overall user experience. This was compounded by the already heavy computational demands of Streamlit, ngrok, and the large AI models.
Future Handling: Direct cloud deployment (as mentioned above) would remove the need for ngrok entirely, providing a much faster and more reliable public interface. Alternatively, for local development, exploring direct IP access within a controlled network could bypass ngrok's overhead.
Model Size and Inference Speed: While TinyLlama is relatively small, the combined inference of multiple models (especially Stable Diffusion) and the RAG pipeline still demanded substantial computational power. The free T4 GPU, while helpful, imposed limitations on generation speed and the complexity of prompts that could be processed efficiently.
Future Handling: Access to more powerful GPUs (e.g., A100, H100) or distributed inference solutions would dramatically accelerate content generation. Optimizing inference code further, exploring ONNX Runtime or TensorRT for model acceleration, and leveraging serverless inference platforms could also improve speed and efficiency.
🛠️ Technologies Used
Python
Hugging Face Transformers: For loading and fine-tuning LLMs, tokenization.
PEFT (Parameter-Efficient Fine-tuning): Specifically LoRA for efficient fine-tuning of TinyLlama.
Diffusers: For image generation with Stable Diffusion.
LangChain: For text splitting and building the RAG pipeline (vector store, embeddings).
FAISS: For efficient similarity search in the RAG pipeline.
Sentence-Transformers: For generating embeddings.
gTTS (Google Text-to-Speech): For audio narration.
Streamlit: For building the interactive web application.
PyTorch: Underlying deep learning framework.
BitsAndBytes: For 8-bit quantization during fine-tuning.
Datasets: For efficient dataset handling.
Pillow: For image manipulation.
Matplotlib: For displaying generated images in notebooks.
Tqdm: For progress bars during dataset generation.
Backoff: For handling API rate limits during dataset generation.
🤝 Contribution
This project is developed by Manoghn.
📄 License
(Consider adding a license, e.g., MIT, Apache 2.0)
