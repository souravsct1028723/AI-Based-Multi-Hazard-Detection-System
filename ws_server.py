import asyncio
import websockets
import sqlite3
import json
import uuid
import time

connected_clients = set()

# =========================
# DATABASE SETUP
# =========================
conn = sqlite3.connect("users.db", check_same_thread=False)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE,
    phone TEXT,
    password TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT,
    confidence REAL,
    location TEXT,
    timestamp INTEGER,
    resolved INTEGER DEFAULT 0
)
""")

conn.commit()


# =========================
# TIMESTAMP NORMALIZER
# =========================
def normalize_timestamp(ts):

    if ts is None:
        return int(time.time() * 1000)

    ts = int(ts)

    if ts < 1000000000000:
        ts = ts * 1000

    return ts


# =========================
# BROADCAST ALERTS
# =========================
async def broadcast(message):

    dead_clients = []

    for client in connected_clients:
        try:
            await client.send(message)
        except:
            dead_clients.append(client)

    for client in dead_clients:
        connected_clients.discard(client)


# =========================
# WEBSOCKET HANDLER
# =========================
async def handler(websocket):

    print("Client connected")
    connected_clients.add(websocket)

    try:

        async for message in websocket:

            print("Received:", message)

            try:
                data = json.loads(message)
            except:
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": "Invalid JSON"
                }))
                continue


            # =========================
            # REGISTER USER
            # =========================
            if data["type"] == "register":

                try:
                    cursor.execute(
                        "INSERT INTO users (email, phone, password) VALUES (?, ?, ?)",
                        (data["email"], data["phone"], data["password"])
                    )

                    conn.commit()

                    await websocket.send(json.dumps({
                        "status": "ok"
                    }))

                except sqlite3.IntegrityError:

                    await websocket.send(json.dumps({
                        "status": "error",
                        "message": "User already exists"
                    }))


            # =========================
            # LOGIN USER
            # =========================
            elif data["type"] == "login":

                cursor.execute(
                    "SELECT * FROM users WHERE email=? AND password=?",
                    (data["email"], data["password"])
                )

                user = cursor.fetchone()

                if user:

                    token = str(uuid.uuid4())

                    await websocket.send(json.dumps({
                        "status": "ok",
                        "token": token
                    }))

                else:

                    await websocket.send(json.dumps({
                        "status": "error",
                        "message": "Invalid email or password"
                    }))


            # =========================
            # FETCH ALERTS
            # =========================
            elif data["type"] == "get_alerts":

                alert_type = data.get("alert_type")

                # Accident alerts
                if alert_type == "accident":

                    cursor.execute("""
                        SELECT * FROM alerts
                        WHERE type='ACCIDENT'
                        ORDER BY timestamp DESC
                    """)

                # Human fall alerts
                elif alert_type == "fall":

                    cursor.execute("""
                        SELECT * FROM alerts
                        WHERE type='FALL'
                        ORDER BY timestamp DESC
                    """)

                # Street light faults
                elif alert_type == "light_fault":

                    cursor.execute("""
                        SELECT * FROM alerts
                        WHERE type='BULB_FAULT' OR type='TWINKLING_FAULT'
                        ORDER BY timestamp DESC
                    """)

                # All alerts
                else:

                    cursor.execute("""
                        SELECT * FROM alerts
                        ORDER BY timestamp DESC
                    """)

                rows = cursor.fetchall()

                alerts = []

                for row in rows:

                    alerts.append({
                        "id": row["id"],
                        "type": row["type"],
                        "confidence": row["confidence"],
                        "location": row["location"],
                        "timestamp": row["timestamp"],
                        "resolved": row["resolved"]
                    })

                await websocket.send(json.dumps({
                    "status": "ok",
                    "alerts": alerts
                }))


            # =========================
            # RESOLVE ALERT
            # =========================
            elif data["type"] == "resolve_alert":

                alert_id = data.get("id")

                if alert_id is not None:

                    cursor.execute(
                        "UPDATE alerts SET resolved=1 WHERE id=?",
                        (alert_id,)
                    )

                    conn.commit()

                    await websocket.send(json.dumps({
                        "status": "ok",
                        "message": "Alert resolved"
                    }))

                else:

                    await websocket.send(json.dumps({
                        "status": "error",
                        "message": "Invalid alert ID"
                    }))


            # =========================
            # AI ALERTS FROM AI SYSTEM
            # =========================
            elif data["type"] in [
                "ACCIDENT",
                "FALL",
                "BULB_FAULT",
                "TWINKLING_FAULT"
            ]:

                timestamp = normalize_timestamp(
                    data.get("timestamp")
                )

                cursor.execute("""
                    INSERT INTO alerts
                    (type, confidence, location, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (
                    data["type"],
                    data.get("confidence", 0),
                    data.get("location", "Unknown"),
                    timestamp
                ))

                conn.commit()

                alert_payload = {
                    "type": data["type"],
                    "confidence": data.get("confidence", 0),
                    "location": data.get("location", "Unknown"),
                    "timestamp": timestamp,
                    "resolved": 0
                }

                print("AI ALERT STORED:", alert_payload)

                await broadcast(json.dumps({
                    "type": "new_alert",
                    "alert": alert_payload
                }))


            # =========================
            # UNKNOWN REQUEST
            # =========================
            else:

                await websocket.send(json.dumps({
                    "status": "error",
                    "message": "Unknown request type"
                }))


    except websockets.exceptions.ConnectionClosed:

        print("Client disconnected")

    finally:

        connected_clients.discard(websocket)


# =========================
# MAIN SERVER
# =========================
async def main():

    async with websockets.serve(handler, "localhost", 8765):

        print("WebSocket Server Running at ws://localhost:8765")

        await asyncio.Future()


asyncio.run(main())