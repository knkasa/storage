# Example of using openai codex.
# https://www.youtube.com/watch?v=Ru5fQZ714x8

import os
import openai

openai.api_key = "sk-COiNTQmY9rqXuxxxxxxxxxxxxxx"  

print("What is your questions?")
user_response = input()

chat_response = openai.Completion.create(
                                    engine = "text-davinci-003",
                                    prompt = user_response,
                                    temperature = 0.5,  # controls randomness in the chat answer.  
                                    max_tokens = 100,
                                    )
                                    
print( chat_response.choices[0].text ) 
                                    


