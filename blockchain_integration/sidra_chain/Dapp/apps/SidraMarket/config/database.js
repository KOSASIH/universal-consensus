const mongoose = require('mongoose');

const databaseConfig = {
  url: 'mongodb://localhost:27017/mydatabase',
  options: {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  },
};

mongoose.connect(databaseConfig.url, databaseConfig.options);

module.exports = mongoose;
