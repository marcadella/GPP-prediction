import os
import http.client, urllib
from IPython import get_ipython
from IPython.core.ultratb import AutoFormattedTB

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


def jupyter_preprocess():
    """Function to run at the start of Jupyter Notebook."""
    # ref. https://stackoverflow.com/a/40135960
    itb = AutoFormattedTB(mode="Plain", tb_offset=1)

    def custom_exc(shell, etype, evalue, tb, tb_offset=None):
        shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
        stb = itb.structured_traceback(etype, evalue, tb)
        sstb = itb.stb2text(stb)

        # Write the code to be executed during an exception here
        ding("ERROR: An exception has occurred.")

        return sstb

    get_ipython().set_custom_exc((Exception,), custom_exc)