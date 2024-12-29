document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    const promptButtons = document.querySelectorAll('.prompt-btn');

    // Add click handlers for example prompt buttons
    promptButtons.forEach(button => {
        button.addEventListener('click', function () {
            const promptText = this.textContent.replace(/['"]/g, ''); // Remove quotes
            userInput.value = promptText;
            form.dispatchEvent(new Event('submit'));
        });
    });

    form.addEventListener('submit', function (e) {
        e.preventDefault();
        const message = userInput.value;
        if (!message.trim()) return; // Prevent empty messages

        appendMessage(message, 'user-message');
        userInput.value = '';

        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        })
            .then(response => response.json())
            .then(data => {
                appendMessage(data.response, 'bot-message');
            })
            .catch(error => console.error('Error:', error));
    });

    function appendMessage(message, className) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${className}`;
        if (className === 'bot-message') {
            messageDiv.innerHTML = message;
        } else {
            messageDiv.textContent = message;
        }
        chatBox.appendChild(messageDiv);

        chatBox.scrollTo({
            top: chatBox.scrollHeight,
            behavior: "smooth"
        });
    }
});
