const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// ONNX model dosyasinin require(...) ile uygulama asset'i olarak paketlenmesini saglar.
config.resolver.assetExts.push('onnx');

module.exports = config;
