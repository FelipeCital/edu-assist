
# streamlit_app.py

import streamlit as st

from transformers import pipeline

from topic_identification import build_lda_model
from video_to_audio_to_text import convert_video_to_audio, audio_to_text, save_text
from text_summarization import (
    extract_text_from_pdf,
    extract_text_from_docx,
    text_rank_summarize,
    extract_keywords,
    display_named_entities,
    translate_text,
    detect_language,
    MODELS
)

#Load CSS
with open("styles.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#RagChatBot
qa_pipeline = pipeline("question-answering")


def video_transcription_page():
    # File uploader allows user to add their own video
    uploaded_video = st.file_uploader(" ", type=["mp4", "avi", "mov"])

    # Only proceed if a video has been uploaded
    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        temp_video_file = "temp_video.mp4"
        try:
            with open(temp_video_file, "wb") as f:
                f.write(uploaded_video.read())
            # Commented out the success message for uploading to keep the UI clean
            # st.success("Video successfully uploaded and saved to temporary file.")
        except Exception as e:
            st.error(f"Failed to save the uploaded video: {e}")
            return
        
        # Convert video to audio and transcribe
        try:
            audio_file_path = convert_video_to_audio(temp_video_file)
            # Commented out the success message for conversion to keep the UI clean
            # st.success(f"Video converted to audio: {audio_file_path}")
        except Exception as e:
            st.error(f"Failed to convert video to audio: {e}")
            return

        try:
            transcribed_text = audio_to_text(audio_file_path)
            if transcribed_text:
                # st.success("Audio successfully transcribed.")  # Also commented out
                st.text_area("Transcribed Text", transcribed_text, height=300)

                # Below is your detailed message about the transcription. This can stay as it provides user guidance.
                st.markdown("""
### Your Video Transcribed
Above is the text extracted from your video. Now you can:

- **Read**: Go through the content at your own pace.
- **Search**: Find key terms using the search function (Ctrl+F or Cmd+F).
- **Save**: Keep a text copy for reference or note-taking.

Having this text means you can revisit complex ideas anytime, anywhere—without needing to hit play.  
  
And if there's something you're not quite clear on, just ask our RAG chatbot for a more in-depth explanation.
""")
                # Assuming save_text is defined elsewhere and works without errors. 
                # This step can be made optional or removed to keep the UI focused on transcription results.
                output_file_path = temp_video_file.rsplit('.', 1)[0] + '_transcribed_text.txt'
                save_text(transcribed_text, output_file_path)
                # st.success(f"Transcribed text saved to {output_file_path}")  # Consider removing or making this optional based on user action
            else:
                st.error("Transcription failed. No text was returned.")
        except Exception as e:
            st.error(f"Failed to transcribe audio: {e}")


def main():

    st.sidebar.title("")
    selection = st.sidebar.radio("", ["Home", "Summarize My Text", "Discover Text Topics", "Convert Video to Text", "Get Instant Answers"])
    
    if selection == "Home":
        st.title("Welcome to EduAssist: Your Academic Companion")

        st.markdown("""
Hello, students! Welcome to EduAssist, the one-stop solution designed to make your academic journey smoother and more efficient. With our suite of tools, you can easily transform your study materials into concise summaries, identify key topics, convert educational videos into text, and interact with our smart chatbot for instant knowledge retrieval and understanding.

**Why EduAssist?**
- **Save Time**: Summarize lengthy documents and videos to grasp essential content in minutes.
- **Study Smarter**: Identify key topics and themes in your study materials to focus on what matters most.
- **Accessible Learning**: Convert educational videos into text for easier review and study.
- **Instant Help**: Our RAG chatbot is here to answer your questions, help with research, and guide you through complex topics.

**Getting Started:**
1. **Choose a Tool**: Select from Text Summarization, Topic Identification, Video to Text, or RAG Chatbot.
2. **Upload Your Files**: Easily upload your lecture notes, documents, or videos.
3. **Get Results**: Receive instant, actionable insights to aid your studies.

EduAssist is here to support your learning journey by providing quick, accessible, and effective study aids. Dive in and discover how you can elevate your studying, understand complex topics with ease, and save precious time!

**Quick Access to RAG Chatbot:**
For instant help or quick answers to your questions, the RAG chatbot is available at the bottom of every selection you make from the navigation bar. It's there to provide you with faster results and support whenever you need it.

**Let's Get Started!**
""")
    
    elif selection == "Summarize My Text":
        st.header("Summarize My Text")
        # Instructions
        st.markdown("""
Struggling with loads of readings? Let's break it down together!  
Just upload your reading material (text, PDF, or DOCX), and in a snap, you'll get a concise summary, the key points, and important terms highlighted for quick understanding.  
Plus, translate it into another language if you need to!
""")
        
        if 'summary' not in st.session_state:
            st.session_state['summary'] = None
        if 'keywords' not in st.session_state:
            st.session_state['keywords'] = None
        if 'entities' not in st.session_state:
            st.session_state['entities'] = None

        text_input = st.text_area("Paste text here:", height=100)
        uploaded_file = st.file_uploader("Or upload a file (.txt, .pdf, .docx):", type=["txt", "pdf", "docx"])

        # Handle file upload
        if uploaded_file is not None:
            # Extract text from the uploaded file here and assign it to text_input
            if uploaded_file.type == "application/pdf":
                text_input = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "text/plain":
                text_input = str(uploaded_file.read(), "utf-8")
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text_input = extract_text_from_docx(uploaded_file)

        # When the button is clicked, generate the summary and other features and save to session_state
        if st.button("Summarize Text"):
            if text_input:
                language = detect_language(text_input)
                if language and language in MODELS:
                    nlp_model = MODELS[language]
                    st.session_state['summary'] = text_rank_summarize(text_input, nlp_model, n_sentences=3)
                    st.session_state['keywords'] = extract_keywords(nlp_model, text_input)
                    st.session_state['entities'] = display_named_entities(nlp_model, text_input)
                    
                    st.write("Summary:")
                    st.write(st.session_state['summary'])
                    st.write("Keywords:")
                    st.write(", ".join([keyword for keyword, _ in st.session_state['keywords']]))
                    st.write("Named Entities:")
                    st.write(", ".join([f"{text} ({label})" for text, label in st.session_state['entities']]))
                else:
                    st.error("Language not supported or text is too short to detect language.")
            else:
                st.error("Please input text or upload a file to summarize.")

        # Translation of the summarized text
        if st.session_state['summary']:
            if st.checkbox("Translate Summary"):
                target_language = st.selectbox("Select target language:", ["English", "Spanish", "French", "German", "Portuguese"], index=0)
                translation = translate_text(st.session_state['summary'], target_language)
                st.write("Translated Text:")
                st.write(translation)
        else:
            st.write("Please generate a summary before attempting to translate.")
                
    elif selection == "Discover Text Topics":
        st.title("Discover Text Topics")
        
        st.markdown("""
Got a big chunk of text to unpack? We've got you covered!  
Upload your document or paste your text, and with a click, we’ll highlight the main topics for you.  
It's like having a personal assistant to help you pinpoint what's important in your readings!
""")
        documents = []
            # User input for the number of topics
        num_topics = st.number_input("Choose how many main ideas you want to see:", min_value=1, max_value=20, value=2, step=1)
        
        # File upload option
        uploaded_file = st.file_uploader("Upload your document here (.txt):", type=['txt'])
        
        # Text area for direct input
        document_input = st.text_area("Or enter document text here:", height=100)
        if document_input:
            documents.append(document_input)

        if uploaded_file is not None:
            # Read the content of the file
            stringio = uploaded_file.getvalue().decode("utf-8")
            documents.extend(stringio.split('\n'))  # Assuming each line of the file is a separate document

        # Results Explanation
        st.markdown("""
        **Understanding Your Topics:**  
        After clicking 'Discover Topics', you'll see a list of topics identified in your text.  
        Each topic is represented by keywords and their relevance score, which indicates how strongly each word is associated with the topic.  
        A higher score means a closer association to the topic.
        
        Unsure about some of the topics you've found? No problem!  
        Just scroll down to our RAG chatbot, paste any text from your document, and ask away.
        """)

        if st.button("Discover Topics"):
            if not documents:
                st.warning("Please upload a file or paste text to analyze.")
            else:
                # Process the documents to identify topics
                lda_model, id2word, corpus = build_lda_model(documents, num_topics=num_topics)
                
                # Display the identified topics and their top words
                st.write("Identified Topics and Top Words:")
                topics = lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=10)
                for topic in topics:
                    st.write(topic)
                    
            

    elif selection == "Convert Video to Text":
        st.title("Video To Text Transcription")
        st.markdown("""
Drowning in video lectures? Let's turn them into text!

Just upload your video, and we'll extract the speech into written words. It's perfect for reviewing key points later or when you prefer reading over watching. Ready to dive in?

1. **Upload Your Video**: Supports MP4, AVI, MOV, and MPEG formats.
2. **Transcribe**: Hit the 'Transcribe Video' button to start the magic.
3. **Review**: Scroll down to see your video's words in print.

Upload your file below and transform your video lectures into manageable texts!
""")
        
        video_transcription_page()

    elif selection == "Get Instant Answers":
        st.title("Answers to Your Study Questions")
        st.markdown("""
###  RAG Chatbot
Got a question while studying? Type it in below, and our RAG chatbot will provide you with the information you need—like a tutor who's always there for you!

Here’s how to get answers:
1. Paste the text you're studying into the box above.
2. Ask any question about the text in the question box.
3. Hit 'Find Your Answer', and our chatbot will fetch the details you need to understand better and learn faster.

No more confusion, no more unanswered questions. Let's make learning easy and fun!
""")
    
        # Text input
        user_text = st.text_area("RAG chatbot - Share Your Text Snippet:")
        
        # Question input
        user_question = st.text_input("What's Your Question?")
        
        if st.button("Find Your Answer"):
            if user_text and user_question:
                # Generate answer
                result = qa_pipeline(question=user_question, context=user_text)
                st.write("Answer:", result['answer'])
        else:
            st.write("Please provide both the text and a question.")
        

if __name__ == "__main__":
    main()
