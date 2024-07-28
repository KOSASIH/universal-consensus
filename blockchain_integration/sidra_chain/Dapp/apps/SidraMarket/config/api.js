const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');

const apiConfig = {
  port: 3000,
  corsOptions: {
    origin: 'http://localhost:3000',
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    allowedHeaders: ['Content-Type', 'Authorization'],
  },
};

const app = express();

app.use(cors(apiConfig.corsOptions));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

module.exports = app;
