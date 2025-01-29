import random

# Predefined responses for different intents
responses = {
    "greeting": ["Hello! How can I assist you today?", "Hi there! What can I do for you?", "Greetings! How may I help you?"],
    "name_query": ["I am a simple chatbot created to assist you."],
    "help_request": ["Sure! What do you need help with?"],
    "goodbye": ["Goodbye! Have a great day!", "See you later!"],
    "capabilities_query": ["I can answer your questions and provide assistance!"]
}

# Function to display the menu and get user choice
def display_menu():
    print("\nPlease choose an option:")
    print("1. Greet me")
    print("2. Ask my name")
    print("3. Request help")
    print("4. Ask what I can do")
    print("5. Say goodbye")
    print("6. Exit")

# Function to generate a response based on the selected option
def chatbot_response(option):
    if option == "1":
        return random.choice(responses["greeting"])
    elif option == "2":
        return random.choice(responses["name_query"])
    elif option == "3":
        return random.choice(responses["help_request"])
    elif option == "4":
        return random.choice(responses["capabilities_query"])
    elif option == "5":
        return random.choice(responses["goodbye"])
    else:
        return "I'm sorry, I didn't understand that. Can you please rephrase?"

def main():
    print("Welcome to the simple menu-based chatbot!")
    while True:
        display_menu()
        user_input = input("You: ")
        
        if user_input == "6":
            print("Chatbot: Goodbye! Have a great day!")
            break
        
        response = chatbot_response(user_input)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()