const messagesContainer = document.getElementById("messages");

function addMessage(content, sender) {
  const message = document.createElement("div");
  message.classList.add("message", sender);
  message.textContent = content;
  messagesContainer.appendChild(message);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

async function sendMessage() {
  const userInput = document.getElementById("user-input");
  const userMessage = userInput.value;

  if (!userMessage) return;

  // Display user's message
  addMessage(userMessage, "user");
  userInput.value = "";

  // Parse user input into the expected payload
  const inputs = userMessage.split(",").map(item => item.trim()); // Example input: "30, Female, 543"
  const payload = {
    age: parseInt(inputs[0], 10),
    gender: inputs[1],
    purchase_amount: parseFloat(inputs[2]),
  };

  console.log("Payload being sent to backend:", payload); // Debugging

  // Send the message to the backend API
  try {
    const response = await fetch("http://127.0.0.1:5000/chatbot", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload) // Send the correct payload
    });

    if (response.ok) {
      const data = await response.json();
      const botMessage = data.response || "Sorry, I couldn't understand that.";
      addMessage(botMessage, "bot");
    } else {
      console.log("Response not OK:", response); // Debugging
      addMessage("Error: Unable to get a response from the server.", "bot");
    }
  } catch (error) {
    console.error("Error:", error);
    addMessage("Error: Unable to connect to the server.", "bot");
  }
}

function handleKeyPress(event) {
  if (event.key === "Enter") {
    sendMessage();
  }
}
