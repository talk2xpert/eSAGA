# from flask import Flask, request, jsonify, render_template
# import requests
#
# import test
# import test2
#
# app = Flask(__name__)
#
# @app.route('/submit', methods=['GET', 'POST'])
# def index():
#
#     payload_data = request.form.get('message')
#     response = send_payload(payload_data)
#     print(response)
#     # Prepare chat history to send back to template
#     chat_history = [
#         {'speaker': 'You', 'message': payload_data},
#         {'speaker': 'Bot', 'message': response.text}
#     ]
#     return render_template('index.html', chat_history=chat_history)
#
#
# @app.route('/submitT', methods=['GET', 'POST'])
# def indexT():
#     user_input = request.form.get('message')
#
#     if not user_input:
#         return render_template('index.html', chat_history=["Please enter a message."])
#
#     bot_response = test.transformer(user_input)
#
#     # Prepare chat history to send back to template
#     chat_history = [
#         {'speaker': 'You', 'message': user_input},
#         {'speaker': 'Bot', 'message': bot_response}
#     ]
#
#     return render_template('index.html', chat_history=chat_history)
#
# @app.route('/sentiment', methods=['GET', 'POST'])
# def sentiment():
#     user_input = request.form.get('message')
#
#     if not user_input:
#         return render_template('index.html', chat_history=["Please enter a message."])
#
#     bot_response = test2.sentiment_analysis(user_input)
#
#     # Prepare chat history to send back to template
#     chat_history = [
#         {'speaker': 'You', 'message': user_input},
#         {'speaker': 'Bot', 'message': bot_response}
#     ]
#
#     return render_template('index.html', chat_history=chat_history)
#
# @app.route('/', methods=['GET', 'POST'])
# def home():
#     return render_template('index.html')
#
# def send_payload(payload_data):
#     url = f'https://payload.vextapp.com/hook/X0S4M03X26/catch/playground'
#     headers = {
#         'Content-Type': 'application/json',
#         'Apikey': 'Api-Key INAUD6Lb.NB7mOFU8PzHYmAoZWWYcGMTDmaJrcDHR'
#     }
#     data = {
#         "payload": payload_data
#     }
#
#     response = requests.post(url, headers=headers, json=data)
#     print(response)
#     return response
#
#
#
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
