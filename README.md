# Smart Parking Management System
## Step-by-Step Setup Guide

---

## WHAT YOU NEED (Prerequisites)
1. **Node.js** — Download from https://nodejs.org (get the LTS version)
2. **SQL Server** — Either SQL Server Express (free) or full SQL Server
3. **SQL Server Management Studio (SSMS)** — To run the SQL file

---

## FOLDER STRUCTURE
After setup, your project folder should look like this:

```
smart-parking/
├── server.js            ← Backend server
├── package.json         ← Node dependencies
├── database/
│   └── setup.sql        ← Run this in SSMS first
└── public/
    └── index.html       ← Frontend (auto-served by server)
```

---

## STEP 1 — Set Up the Database

1. Open **SQL Server Management Studio (SSMS)**
2. Connect to your SQL Server instance
3. Click **File → Open → File** and open `database/setup.sql`
4. Click **Execute** (or press F5)
5. You should see: "Database setup complete!"

---

## STEP 2 — Configure Database Connection

Open `server.js` in any text editor (Notepad, VS Code, etc.)

Find this section near the top:

```javascript
const dbConfig = {
  server:   'localhost',           // ← Change this
  database: 'SmartParking',
  authentication: {
    type: 'default',
    options: {
      userName: 'sa',              // ← Your SQL login
      password: 'YourPassword123', // ← Your SQL password
    }
  }
};
```

### Common server values:
- If using **SQL Server Express**: change `'localhost'` to `'.\\SQLEXPRESS'`
- If using full SQL Server: keep `'localhost'`
- If your instance has a name like `DESKTOP-ABC\SQLEXPRESS`, use that

### If you use Windows Authentication instead of username/password:
Replace the authentication block with:
```javascript
authentication: {
  type: 'ntlm',
  options: { domain: '', userName: '', password: '' }
},
options: {
  encrypt: false,
  trustServerCertificate: true,
  enableArithAbort: true,
},
```

---

## STEP 3 — Install Node Packages

Open **Command Prompt** or **Terminal** in the `smart-parking` folder:

```
cd path\to\smart-parking
npm install
```

This installs: express, mssql, cors, body-parser

---

## STEP 4 — Run the Server

In the same terminal:

```
node server.js
```

You should see:
```
Connected to SQL Server - SmartParking
Server running at http://localhost:3000
Demo credentials:
  Admin: admin@parking.com / admin123
  User:  user@parking.com  / user123
```

---

## STEP 5 — Open the App

Open your browser and go to:
**http://localhost:3000**

You will see the Login page.

---

## DEMO CREDENTIALS

| Role  | Email                | Password |
|-------|----------------------|----------|
| Admin | admin@parking.com    | admin123 |
| User  | user@parking.com     | user123  |

---

## WHAT EACH ROLE CAN DO

### Admin
- View full dashboard with all stats
- Manage ALL parking slots (add/delete)
- View ALL vehicles, records, payments
- Record vehicle entry/exit
- Set parking rates & pricing
- Manage all users
- View/issue fines
- Manage all reservations

### User
- View dashboard with their own data only
- See their own vehicles
- See their own parking records
- See their own payments
- Make reservations
- View fines on their vehicles

---

## TROUBLESHOOTING

**"Cannot connect to server" on login:**
→ Make sure `node server.js` is running in the terminal

**"DB connection failed" in terminal:**
→ Check your server name, username, and password in server.js
→ Make sure SQL Server is running (check Windows Services)

**SQL Server Express connection string:**
→ Change `server: 'localhost'` to `server: '.\\SQLEXPRESS'`

**Port 3000 already in use:**
→ Change `const PORT = 3000;` at the bottom of server.js to 3001 or any other port
→ Also update `const API = 'http://localhost:3000/api';` in index.html to match
