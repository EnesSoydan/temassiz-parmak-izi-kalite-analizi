# Mobil model dosyaları

Bu klasörde uygulamanın cihaz içi ONNX modelleri bulunur. Model dosyaları boyutları nedeniyle Git tarafından takip edilmez.

Beklenen dosyalar:

```text
fingertip_obb.onnx
fingertip_segmentation.onnx
```

Dosyaları model eğitim/export çıktılarından veya proje sahibinin kullandığı güvenilir model deposundan bu klasöre kopyalayın. Uygulama build’i öncesinde `app.json` ve `src/lib/onnx-model.ts` içindeki dosya adlarıyla birebir eşleştiğini kontrol edin.

Model, dataset, `.pt`, `.onnx`, `runs` ve test fotoğrafları GitHub’a yüklenmemelidir.
