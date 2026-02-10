# templates.py

HTML_LOBBY = """
<html>
<head>
    <title>Flower VLA Mission Control</title>
    <style>
        body { background-color: #1a1a1a; color: white; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { color: #4CAF50; text-align: center; font-size: 2.5em; margin-bottom: 10px; }
        .loading { text-align: center; color: #ffeb3b; font-size: 1.5em; margin-top: 50px; }
        .scene-list { list-style: none; padding: 0; }
        .scene-item { 
            background: #2d2d2d; margin: 8px 0; padding: 18px; border-radius: 8px; cursor: pointer; 
            transition: all 0.2s; display: flex; align-items: center;
            border-left: 5px solid #444;
        }
        .scene-item:hover { background: #3d3d3d; border-left: 5px solid #4CAF50; transform: translateX(5px); }
        .scene-id { font-weight: bold; color: #888; margin-right: 15px; font-family: monospace; font-size: 1.2em; min-width: 40px;}
        .scene-desc { font-size: 1.1em; color: #eee; }
        a { text-decoration: none; color: white; display: block; }
        hr { border-color: #333; margin-bottom: 30px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Flower VLA Mission Control</h1>
        <div style="text-align: center; color: #888; margin-bottom: 30px;">Select a simulation environment to begin</div>
        <hr>
        {% if not ready %}
            <div class="loading">
                <p>⏳ <b>System Initializing...</b></p>
                <p style="font-size: 0.8em; color: #ccc">{{ progress }}</p>
                <div style="margin-top:20px; font-size: 0.6em; color: #666">(This usually takes 3-4 minutes. Page auto-refreshes.)</div>
                <script>setTimeout(function(){ location.reload(); }, 3000);</script>
            </div>
        {% else %}
            <ul class="scene-list">
            {% for id, desc in scenes.items() %}
                <a href="/start/{{ id }}">
                <li class="scene-item">
                    <span class="scene-id">{{ id }}</span>
                    <span class="scene-desc">{{ desc }}</span>
                </li>
                </a>
            {% endfor %}
            </ul>
        {% endif %}
    </div>
</body>
</html>
"""

HTML_STREAM = """
<html>
<head>
    <title>Flower VLA Live Feed</title>
    <style>
        body { background-color: #1a1a1a; color: white; font-family: sans-serif; text-align: center; margin: 0; padding: 20px; }
        h1 { margin-top: 10px; color: #4CAF50; }
        
        img { 
            border: 4px solid #333; 
            border-radius: 5px; 
            width: 95%; 
            min-width: 800px; 
            box-shadow: 0 0 20px rgba(0,0,0,0.5);
            background: #000;
        }
        
        .control-panel { 
            margin-top: 20px; 
            padding: 20px; 
            background: #222; 
            display: inline-block; 
            border-radius: 10px; 
            width: 80%; 
        }
        
        .task-name { color: #fff; font-size: 1.2em; font-weight: bold; margin-bottom: 20px; }
        
        /* Command Input Styling */
        .cmd-group { display: flex; justify-content: center; gap: 10px; margin-bottom: 10px; align-items: center; }
        
        input[type="text"] {
            padding: 15px;
            font-size: 1.2em;
            border-radius: 5px;
            border: 2px solid #555;
            background: #333;
            color: white;
            flex-grow: 2;
        }
        
        /* New Step Counter Input */
        .step-input-group {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }
        .step-label {
            font-size: 0.8em;
            color: #aaa;
            margin-bottom: 3px;
            margin-left: 2px;
        }
        input[type="number"] {
            padding: 15px;
            font-size: 1.2em;
            border-radius: 5px;
            border: 2px solid #555;
            background: #333;
            color: #4CAF50;
            width: 100px;
            font-weight: bold;
            text-align: center;
        }

        button {
            padding: 15px 30px;
            font-size: 1.2em;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover { background: #45a049; }
        button:disabled { background: #555; cursor: not-allowed; }
        
        #status-msg { color: #aaa; margin-top: 10px; font-style: italic; }
    </style>
    <script>
        function sendCommand() {
            const input = document.getElementById("cmd-input");
            const stepsInput = document.getElementById("steps-input");
            const btn = document.getElementById("cmd-btn");
            const status = document.getElementById("status-msg");
            
            const cmd = input.value.trim();
            const steps = parseInt(stepsInput.value) || 400; // Default to 400 if empty
            
            if (!cmd) return;
            
            btn.disabled = true;
            status.innerText = "Sending command (" + steps + " steps)...";
            
            fetch('/submit_command', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    command: cmd,
                    max_steps: steps
                })
            })
            .then(response => response.json())
            .then(data => {
                input.value = "";
                btn.disabled = false;
                status.innerText = "Executing: " + cmd;
                setTimeout(() => { status.innerText = "Ready for next command."; }, 5000);
            })
            .catch(error => {
                console.error('Error:', error);
                btn.disabled = false;
                status.innerText = "Error sending command.";
            });
        }
        
        // Allow Enter key to submit
        document.addEventListener("DOMContentLoaded", function() {
            document.getElementById("cmd-input").addEventListener("keypress", function(event) {
                if (event.key === "Enter") {
                    event.preventDefault();
                    sendCommand();
                }
            });
        });
    </script>
</head>
<body>
    <h1>🤖 Live Operation</h1>
    <img src="/video_feed" />
    
    <div class="control-panel">
        <div class="task-name">{{ desc }}</div>
        
        <div class="cmd-group">
            <input type="text" id="cmd-input" placeholder="Type instruction here (e.g. 'open the drawer')..." autofocus>
            
            <div class="step-input-group">
                <span class="step-label">Max Steps</span>
                <input type="number" id="steps-input" value="400" min="10" max="2000">
            </div>
            
            <button id="cmd-btn" onclick="sendCommand()">🚀 RUN</button>
        </div>
        <div id="status-msg">Enter a command to start the robot.</div>
    </div>
</body>
</html>
"""