const chatBox = document.getElementById("chat-box");
const textInput = document.getElementById("text-input");
const sendBtn = document.getElementById("send-btn");
const micBtn = document.getElementById("mic-button");
const statusText = document.getElementById("status");

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = new SpeechRecognition();
recognition.lang = "en-US";
recognition.interimResults = false;
recognition.continuous = false;

// Handle voice input
micBtn.onclick = () => {
  recognition.start();
  statusText.textContent = "Listening...";
};

recognition.onresult = async (event) => {
  const transcript = event.results[0][0].transcript;
  addMessage("user", transcript);
  textInput.value = "";
  statusText.textContent = "Thinking...";
  const reply = await getResponse(transcript);
  addMessage("assistant", reply);
  speak(reply);
  statusText.textContent = "Type or use your voice to ask a question";
};

// Handle text input
sendBtn.onclick = async () => {
  const query = textInput.value.trim();
  if (!query) return;

  addMessage("user", query);
  textInput.value = "";
  statusText.textContent = "Thinking...";

  const reply = await getResponse(query);
  addMessage("assistant", reply);
  speak(reply);
  statusText.textContent = "Type or use your voice to ask a question";
};

function addMessage(sender, text) {
  const msg = document.createElement("div");
  msg.className = `message ${sender}`;
  msg.textContent = text;
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

// Replace this with your backend API call (e.g., OpenAI, Flask, FastAPI)
async function getResponse(prompt) {
  try {
    const res = await fetch("https://your-api-endpoint.com/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ prompt })
    });

    const data = await res.json();
    return data.response || "Sorry, I didn't understand that.";
  } catch (error) {
    console.error(error);
    return "Failed to connect to assistant.";
  }
}

// Text-to-speech
function speak(text) {
  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = "en-US";
  window.speechSynthesis.speak(utter);
}
