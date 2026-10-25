
import sqlite3
from pathlib import Path

# ==========================================
# DATABASE CONFIGURATION
# ==========================================

DATABASE = Path(__file__).resolve().parent / "chatbot.db"


# ==========================================
# 1. CREATE DATABASE AND TABLES
# ==========================================

def create_database():

    with sqlite3.connect(DATABASE) as connection:

        cursor = connection.cursor()

        # Orders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                customer_name TEXT NOT NULL,
                product TEXT NOT NULL,
                status TEXT NOT NULL
            )
        """)

        # Chat history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Support tickets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS support_tickets (
                ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                issue TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Open',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Sample orders
        orders = [
            ("ORD101", "Rahul", "Laptop", "Shipped"),
            ("ORD102", "Arun", "Headphones", "Delivered"),
            ("ORD103", "Priya", "Mobile Phone", "Processing"),
            ("ORD104", "Kumar", "Keyboard", "Out for Delivery")
        ]

        cursor.executemany("""
            INSERT OR IGNORE INTO orders
            (order_id, customer_name, product, status)
            VALUES (?, ?, ?, ?)
        """, orders)

        connection.commit()


# ==========================================
# 2. TRACK CUSTOMER ORDER
# ==========================================

def track_order(order_id):

    with sqlite3.connect(DATABASE) as connection:

        cursor = connection.execute("""
            SELECT product, status
            FROM orders
            WHERE order_id = ?
        """, (order_id.strip().upper(),))

        order = cursor.fetchone()

    if order:
        return {
            "found": True,
            "product": order[0],
            "status": order[1]
        }

    return {
        "found": False,
        "message": "Order not found."
    }


# ==========================================
# 3. SAVE CHAT HISTORY
# ==========================================

def save_chat(session_id, user_message, bot_response):

    with sqlite3.connect(DATABASE) as connection:

        connection.execute("""
            INSERT INTO chat_history
            (session_id, user_message, bot_response)
            VALUES (?, ?, ?)
        """, (
            session_id,
            user_message,
            bot_response
        ))

        connection.commit()


# ==========================================
# 4. GET CHAT HISTORY
# ==========================================

def get_chat_history(session_id):

    with sqlite3.connect(DATABASE) as connection:

        cursor = connection.execute("""
            SELECT user_message, bot_response, created_at
            FROM chat_history
            WHERE session_id = ?
            ORDER BY id ASC
        """, (session_id,))

        rows = cursor.fetchall()

    history = []

    for row in rows:
        history.append({
            "user": row[0],
            "bot": row[1],
            "time": row[2]
        })

    return history


# ==========================================
# 5. CREATE CUSTOMER SUPPORT TICKET
# ==========================================

def create_ticket(session_id, issue):

    with sqlite3.connect(DATABASE) as connection:

        cursor = connection.execute("""
            INSERT INTO support_tickets
            (session_id, issue)
            VALUES (?, ?)
        """, (
            session_id,
            issue
        ))

        ticket_id = cursor.lastrowid

        connection.commit()

    return ticket_id


# ==========================================
# 6. GET CUSTOMER SUPPORT TICKETS
# ==========================================

def get_tickets(session_id):

    with sqlite3.connect(DATABASE) as connection:

        cursor = connection.execute("""
            SELECT ticket_id, issue, status, created_at
            FROM support_tickets
            WHERE session_id = ?
            ORDER BY ticket_id DESC
        """, (session_id,))

        rows = cursor.fetchall()

    tickets = []

    for row in rows:
        tickets.append({
            "ticket_id": row[0],
            "issue": row[1],
            "status": row[2],
            "created_at": row[3]
        })

    return tickets


# ==========================================
# 7. TEST DATABASE
# ==========================================

if __name__ == "__main__":

    create_database()

    print("Database created successfully!")

    # Test order tracking
    print("\nORDER TRACKING:")
    print(track_order("ORD101"))

    # Test chat history
    save_chat(
        "demo",
        "Hello",
        "Hello! How can I help you?"
    )

    print("\nCHAT HISTORY:")
    print(get_chat_history("demo"))

    # Test support ticket
    ticket_id = create_ticket(
        "demo",
        "My headphones arrived damaged."
    )

    print("\nSUPPORT TICKET CREATED:")
    print("Ticket ID:", ticket_id)

    print("\nALL SUPPORT TICKETS:")
    print(get_tickets("demo"))
