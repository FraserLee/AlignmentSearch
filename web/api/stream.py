from flask import Flask, Response
import time

app = Flask(__name__)

# server side event: stream "hello world" every 2 seconds
@app.route("/api/stream")
def stream():

    def eventStream():
        for i in range(30):
            # yield "data: AAAA(" + str(i) + ")\n\n"
            yield "event: message\ndata: " + str(i) + "\n\n"
            time.sleep(0.1)

    return Response(eventStream(), mimetype="text/event-stream")

