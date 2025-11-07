# Database Setup for Multi-Chat Feature

## Overview

The Engineer Assistant now uses a database (PostgreSQL or SQLite) to persist chat conversations. This allows you to:
- Create multiple named conversations
- Switch between conversations seamlessly
- Automatically save all messages and agent state
- Load previous conversations when you restart the app

## Quick Start (SQLite - Default)

By default, the application uses **SQLite** which requires no setup. Just run the app and it will automatically create a local database file at `data/conversations.db`.

```bash
streamlit run src/ui/streamlit_app.py
```

That's it! Your conversations are automatically saved locally.

## Using PostgreSQL (Optional)

For production use or when you need to share conversations across multiple users, you can use PostgreSQL.

### 1. Install PostgreSQL

**macOS (using Homebrew):**
```bash
brew install postgresql@15
brew services start postgresql@15
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

**Windows:**
Download and install from [postgresql.org](https://www.postgresql.org/download/windows/)

### 2. Create Database

```bash
# Connect to PostgreSQL
psql postgres

# Create database and user
CREATE DATABASE engineer_assistant;
CREATE USER engiai_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE engineer_assistant TO engiai_user;

# Exit psql
\q
```

### 3. Configure Environment Variable

Create or update your `.env` file:

```bash
# Copy example file
cp .env.example .env

# Edit .env and set:
DATABASE_URL=postgresql://engiai_user:your_password@localhost:5432/engineer_assistant
```

### 4. Install Dependencies

```bash
pip install -e .
```

This will install `psycopg2-binary` and `sqlalchemy` from `pyproject.toml`.

### 5. Run the Application

```bash
streamlit run src/ui/streamlit_app.py
```

The database tables will be created automatically on first run.

## Database Schema

The application creates three tables:

### 1. `conversations`
Stores conversation metadata:
- `id` (String/UUID): Primary key, session ID
- `name` (String): Conversation name
- `created_at` (DateTime): Creation timestamp
- `updated_at` (DateTime): Last update timestamp
- `message_count` (Integer): Number of messages

### 2. `messages`
Stores individual messages:
- `id` (Integer): Auto-increment primary key
- `conversation_id` (String): Foreign key to conversations
- `role` (String): 'user' or 'assistant'
- `content` (Text): Message content
- `created_at` (DateTime): Creation timestamp

### 3. `conversation_states`
Stores LangGraph agent state:
- `conversation_id` (String): Primary key, foreign key to conversations
- `agent_state` (JSON): Agent messages and state
- `config` (JSON): LangGraph configuration
- `waiting_for_confirmation` (Integer): Boolean flag (0 or 1)
- `updated_at` (DateTime): Last update timestamp

## How It Works

1. **Create Conversation**: User enters a name in the sidebar and clicks "Create New Conversation"
   - Generates a unique UUID as session_id
   - Creates entry in `conversations` table
   - Initializes empty state in `conversation_states`

2. **Send Message**: User types a message
   - Message saved to `messages` table
   - Agent processes and responds
   - Response saved to `messages` table
   - Agent state saved to `conversation_states`
   - Conversation `updated_at` timestamp updated

3. **Switch Conversation**: User selects a different conversation
   - Current conversation state saved to database
   - Selected conversation loaded from database
   - Messages displayed in chat interface

4. **Load on Startup**: When app starts
   - All conversations loaded from database
   - Most recently updated conversation becomes active
   - Messages and state restored

## Environment Variables

```bash
# SQLite (default - no setup required)
DATABASE_URL=sqlite:///data/conversations.db

# PostgreSQL (local)
DATABASE_URL=postgresql://username:password@localhost:5432/database_name

# PostgreSQL (remote - e.g., Heroku, AWS RDS)
DATABASE_URL=postgresql://username:password@host:port/database_name

# PostgreSQL (with SSL)
DATABASE_URL=postgresql://username:password@host:port/database_name?sslmode=require
```

## Migrating from Pickle Files

If you were using the previous version with pickle files (`data/chats/*.pkl`), those files are no longer used. To migrate:

1. **Option 1**: Start fresh (conversations will be empty)
   - Just start using the new version
   - Old pickle files can be deleted

2. **Option 2**: Manual migration (if you need old conversations)
   - This requires writing a custom migration script
   - Contact support or check for migration tools in the repository

## Backup and Recovery

### SQLite Backup

```bash
# Backup
cp data/conversations.db data/conversations.backup.db

# Restore
cp data/conversations.backup.db data/conversations.db
```

### PostgreSQL Backup

```bash
# Backup
pg_dump -U engiai_user engineer_assistant > backup.sql

# Restore
psql -U engiai_user engineer_assistant < backup.sql
```

## Troubleshooting

### "No module named 'psycopg2'"

```bash
pip install psycopg2-binary
```

### "could not connect to server"

Check that PostgreSQL is running:
```bash
# macOS
brew services list

# Linux
sudo systemctl status postgresql

# Check if port is open
psql -U postgres -h localhost -p 5432
```

### "relation does not exist"

Tables are created automatically. If you get this error:
1. Ensure the database user has CREATE privileges
2. Try manually creating tables by running the app once
3. Check the database URL is correct

### Performance Issues

For SQLite:
- Consider using PostgreSQL for production
- SQLite is great for development but may be slower with many conversations

For PostgreSQL:
- Add indexes if needed (not required for typical usage)
- Consider connection pooling for high traffic

## Production Deployment

### Using Heroku

1. Add Heroku Postgres addon
2. Heroku automatically sets `DATABASE_URL`
3. Deploy your app

### Using Railway/Render

1. Create PostgreSQL database
2. Copy connection string to `DATABASE_URL`
3. Deploy app

### Using Docker

```dockerfile
# In Dockerfile
ENV DATABASE_URL=postgresql://user:pass@db:5432/engineer_assistant

# In docker-compose.yml
services:
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: engineer_assistant
      POSTGRES_USER: engiai_user
      POSTGRES_PASSWORD: your_password
```

## Security Notes

⚠️ **Important Security Considerations:**

1. **Never commit `.env` file** - It contains sensitive credentials
2. **Use strong passwords** for PostgreSQL
3. **Use SSL/TLS** for remote PostgreSQL connections
4. **Restrict database access** - Only allow connections from your app server
5. **Regular backups** - Schedule automated backups of your database
6. **Environment-specific configs** - Use different databases for dev/staging/prod

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the database schema section above
3. Open an issue on GitHub with error logs
