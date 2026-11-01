/**
 * PhantomFlow Conversation Engine Server
 * Express.js entrypoint providing REST endpoints for Member 1 & clients.
 */

const express = require('express');
const cors = require('cors');
const path = require('path');
const config = require('./config/config');
const mongoManager = require('./database/mongoClient');
const conversationRoutes = require('./routes/conversationRoutes');
const speechRoutes = require('./routes/speechRoutes');

const app = express();

// Middlewares
app.use(cors({ origin: config.corsOrigin }));
app.use(express.json());

// Request logging middleware (lightweight)
app.use((req, res, next) => {
  const start = Date.now();
  res.on('finish', () => {
    const duration = Date.now() - start;
    if (req.path !== '/health') {
      console.log(`[HTTP] ${req.method} ${req.path} -> ${res.statusCode} (${duration}ms)`);
    }
  });
  next();
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'ok',
    service: 'phantomflow-conversation-engine',
    version: '1.0.0',
    member: 'Member 3 - AI Conversation & Backend Engineer',
    uptime: Math.floor(process.uptime()),
    providers: config.getPublicStatus().providers,
    timestamp: new Date().toISOString(),
  });
});

// API Routes
app.use('/api/conversation', conversationRoutes);
app.use('/api/speech', speechRoutes);

// Static Web Application for Testing & Voice Services
app.use(express.static(path.join(__dirname, '../public')));

// 404 Handler
app.use((req, res) => {
  res.status(404).json({
    success: false,
    error: `Route not found: ${req.method} ${req.originalUrl}`,
  });
});

// Global Error Handler
app.use((err, req, res, next) => {
  console.error('[Server Error]', err);
  res.status(500).json({
    success: false,
    error: 'Internal Server Error',
    message: err.message,
  });
});

// Start listening if run directly
if (require.main === module) {
  const server = app.listen(config.port, () => {
    console.log('='.repeat(65));
    console.log(`  PHANTOMFLOW Conversation Engine listening on port ${config.port}`);
    console.log(`  Environment: ${config.env}`);
    console.log(`  Health Check: http://localhost:${config.port}/health`);
    console.log(`  API Endpoint: POST http://localhost:${config.port}/api/conversation/message`);
    console.log('='.repeat(65));

    if (config.mongodb.isConfigured) {
      mongoManager.connect().catch((err) => {
        console.warn(`[Server] MongoDB startup note: ${err.message}`);
      });
    }
  });

  // Graceful shutdown handling
  const shutdown = async () => {
    console.log('\nShutting down gracefully...');
    await mongoManager.close();
    server.close(() => {
      console.log('Server closed.');
      process.exit(0);
    });
  };

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

module.exports = app;
