import os
import openai
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from elevenlabs import generate
from streamlit_chat import message
from dotenv import load_dotenv

load_dotenv()

TempAudioFile = "temp_audio.wav"
audioformat = "audio/wav"


# Load environment variables from .env file and return the keys
openai.api_key = os.environ.get('OPENAI_API_KEY')
eleven_api_key = os.environ.get('ELEVEN_API_KEY')
active_loop_data_set_path = os.environ.get('DEEPLAKE_DATASET_PATH')

def load_embeddings(dataset_path):
    embeddings = OpenAIEmbeddings()
    db = DeepLake(dataset_path=dataset_path, read_only=True, embedding_function=embeddings)

    return db



def transcribe_audio_to_text(audio_file):
    try:
        with open(audio_file, "rb") as f:
            response = openai.Audio.transcribe(
                model="whisper-1",
                file=f,
                response_format="text"
            )
        # print(type(response, " response"))
        return response
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None


def record_and_transcribe():
    audioBytes = audio_recorder()
    transcription = None
    if audioBytes:
        st.audio(audioBytes, format=audioformat)
        
        with open(TempAudioFile, "wb") as f:
            f.write(audioBytes)

        if st.button("Transcribe"):
            print("Transcribing audio... type: ", type(TempAudioFile))
            transcription = transcribe_audio_to_text(TempAudioFile)
            os.remove(TempAudioFile)
            display_transcription(transcription)

    return transcription


def display_transcription(transcription):
    if transcription:
        st.write(f"transaction: {transcription}")
        with open("audioTranscriopt.txt", "w+") as f:
            f.write(transcription)
    else:
        st.write("error transcribing audio")


def get_user_text(transcription):
    return st.text_input("", value=transcription if transcription else "", key=input)


def search_db(user_input, db):
    print(user_input)
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 100
    retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = 2
    model = ChatOpenAI(model='gpt-3.5-turbo')
    qa = RetrievalQA.from_llm(model, retriever=retriever, return_source_documents=True)
    return qa({'query': user_input})


def display_conversation(history):

    for i in range(len(history['generated'])):
        message(history['past'][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))

        voice = "Bella"
        text= history["generated"][i]
        audio = generate(text=text, voice=voice)
        st.audio(audio, format='audio/mp3')


# Main function to run the app
def main():
    # Initialize Streamlit app with a title
    st.write("# CharanBase 🧙")
   
    # Load embeddings and the DeepLake database
    db = load_embeddings(active_loop_data_set_path)
    # Record and transcribe audio
    transcription = record_and_transcribe()

    # Get user input from text input or audio transcription
    user_input = get_user_text(transcription)

    # Initialize session state for generated responses and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]
        
    # Search the database for a response based on user input and update session state
    if user_input:
        output = search_db(user_input, db)
        print(len(output["source_documents"]))
        # print(output['source_documents'])
        st.session_state.past.append(user_input)
        response = str(output["result"])
        st.session_state.generated.append(response)

    # Display conversation history using Streamlit messages
    if st.session_state["generated"]:
        display_conversation(st.session_state)

# Run the main function when the script is executed
if __name__ == "__main__":
    main()





