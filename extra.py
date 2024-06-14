import os
import http.client, urllib

# Send a push message via Pushover
def send_push(message):
    try:
        conn = http.client.HTTPSConnection("api.pushover.net:443")
        conn.request(
            "POST",
            "/1/messages.json",
            urllib.parse.urlencode(
                {
                    "token": os.environ.get('PUSHOVER_JUPYTERLAB_TOKEN'),
                    "user": os.environ.get('PUSHOVER_USER'),
                    "message": message,
                }
            ),
            {"Content-type": "application/x-www-form-urlencoded"},
        )
        errcode = conn.getresponse().getcode()
        if errcode >= 400:
            print(f"Could not send push notification (error {errcode})")
    except Exception as e:
        print(f"Could not send push notification ({e})")


# Emit a vocal notification
# Found here: https://mindtrove.info/jupyter-tidbit-run-and-say-done/
def speak(text):
    from IPython.display import Javascript as js, clear_output
    # Escape single quotes
    text = text.replace("'", r"\'")
    display(js('''
    if(window.speechSynthesis) {{
        var synth = window.speechSynthesis;
        synth.speak(new window.SpeechSynthesisUtterance('{text}'));
    }}
    '''.format(text=text)))
    # Clear the JS so that the notebook doesn't speak again when reopened/refreshed
    # Commenting this as it causes the printout before calling this function to not be displayed
    #clear_output(False)

    
def ding(text="Task complete"):
    #speak(text)
    send_push(text)