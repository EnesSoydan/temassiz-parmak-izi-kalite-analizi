import {
  Images,
  type Image as NitroImage,
} from 'react-native-nitro-image';

import {
  createAlignedCanonicalFingerRoiImageUri,
  createBinaryFingerRoiImageUri,
  createEnhancedFingerRoiImageUri,
  createMaskFingerRoiImageUri,
  createMinutiaeFingerRoiImageUri,
  createMinutiaeOverlayFingerRoiImageUri,
  createOpenedBinaryFingerRoiImageUri,
  createOrientationFingerRoiImageUri,
  createSegmentedFingerRoiImageUri,
  saveCanonicalFingerRoiImage,
  saveFingerRoiImage,
} from '@/lib/capture-storage';
import {
  createEnhancedSegmentedPixels,
  rotateRgbaAndMask,
  saveMaskPng,
  saveRgbaImage,
  saveRgbaPng,
  segmentFingerRoiImage,
} from '@/lib/finger-segmentation';
import {
  calculateRidgeScaleFactor,
  rescaleRgbaAndMask,
} from '@/lib/ridge-scale';
import {
  analyzeFingerprintQualityDetailed,
  createSegmentationFailureQuality,
} from '@/lib/fingerprint-quality';
import { sortFingerRois } from '@/lib/finger-order';
import {
  getCanonicalGeometry,
  normalizeAxisRotation,
} from '@/lib/roi-geometry';
import {
  extractFingerprintMinutiae,
  createDistalMinutiaeMask,
  type MinutiaeExtractionResult,
} from '@/lib/minutiae-extraction';
import { saveRidgeEnhancedImage } from '@/lib/ridge-enhancement';
import { createRidgeTextureDescriptor } from '@/lib/texture-descriptor';
import type {
  CaptureSample,
  DetectedObbBox,
  FingerRoi,
  FingerprintQuality,
  MinutiaeRejectionReason,
} from '@/types/biometrics';

type ImageSize = NonNullable<CaptureSample['rawImageSize']>;
// Çok küçük hatalı kırpmaların dosya üretmesini engelleyen minimum piksel boyutu.
const MIN_ROI_PIXEL_SIZE = 16;

// Üç işçi native beklemeleri paylaşırken dördüncü büyük piksel çalışma kopyasını açmayarak belleği sınırlar.
const ROI_PROCESSING_CONCURRENCY = 2;

// Enrollment öncesi geçici template kapısı, çok az veya taşmış aday listesini biyometrik veri olarak saklamaz.
const MIN_TEMPLATE_MINUTIAE_COUNT = 8;
const MAX_TEMPLATE_MINUTIAE_COUNT = 60;
const MIN_SEARCHABLE_AREA_RATIO = 0.015;
const MIN_LARGEST_SEARCHABLE_REGION_RATIO = 0.008;
// Gabor destek alanının sınırındaki orta güvenli adayları da kontrollü biçimde template'e alır.
const MINUTIAE_SUPPORT_CONFIDENCE_FLOOR = 70;

// Canlı veya fotoğraf model kutularından parmak başına normalize ROI planı üretir.
export function createFingerRoiPlans(
  detections: DetectedObbBox[],
  imageSize: ImageSize
): FingerRoi[] {
  return sortFingerRois(
    detections
      .map((detection) => createFingerRoiPlan(detection, imageSize))
      .filter((roi): roi is FingerRoi => roi !== null)
  );
}

// Ham görselden model tespitlerine göre her parmak için ayrı ROI JPEG dosyası çıkarır.
export async function extractFingerRoisFromImage({
  imageUri,
  imageSize,
  detections,
  sampleId,
  sourceImage,
}: {
  imageUri: string;
  imageSize: ImageSize;
  detections: DetectedObbBox[];
  sampleId: string;
  sourceImage?: NitroImage;
}) {
  const roiPlans = createFingerRoiPlans(detections, imageSize);
  const processingStartedAt = Date.now();
  // Kamera çekiminde bellekteki yönlü görüntüyü, galeri akışında ise dosyadan yüklenen tek native görüntüyü paylaşırız.
  const sharedSourceImage =
    sourceImage ?? (await Images.loadFromFileAsync(toNativeFilePath(imageUri)));

  // İki işçi native kırpma ve dosya erişimini üst üste bindirir; üç işçi cihazda JS sıkışmasına yol açıyordu.
  const fingerRois = await mapWithConcurrency(
    roiPlans,
    ROI_PROCESSING_CONCURRENCY,
    async (roiPlan) => {
      const roiSourceImage = sharedSourceImage;
      const fingerStartedAt = Date.now();
      const cropStartedAt = Date.now();

      // Nitro Image aynı çözülmüş fotoğraftan eksene hizalı ham ROI'yi native tarafta kırpar.
      const cropped = await roiSourceImage.cropAsync(
        roiPlan.crop.originX,
        roiPlan.crop.originY,
        roiPlan.crop.originX + roiPlan.crop.width,
        roiPlan.crop.originY + roiPlan.crop.height
      );
      const croppedPath = await cropped.saveToTemporaryFileAsync('jpg', 95);
      const cropMs = Date.now() - cropStartedAt;

      // Geçici ROI dosyasını uygulamanın kalıcı capture klasörüne taşınabilir isimle kopyalar.
      const saveStartedAt = Date.now();
      const savedImageUri = await saveFingerRoiImage(
        toFileUri(croppedPath),
        sampleId,
        roiPlan.className
      );
      const canonical = await createCanonicalFingerRoi(cropped, roiPlan);
      const canonicalPath = await canonical.image.saveToTemporaryFileAsync('jpg', 95);
      const canonicalImageUri = await saveCanonicalFingerRoiImage(
        toFileUri(canonicalPath),
        sampleId,
        roiPlan.className
      );
      const saveMs = Date.now() - saveStartedAt;
      // Aynı sınıf iki karede de varsa küçük native tamponları doğrudan karşılaştırır, ikinci JPEG decode etmez.
      const processedRoi = await tryProcessFingerRoi({
        // Perspektif düzeltmesi OBB köşeleriyle ham eksen ROI'si üzerinde yapılır; native döndürme yalnızca debug önizlemesidir.
        roiImageUri: savedImageUri,
        sampleId,
        className: roiPlan.className,
        sourceWidth: canonical.sourcePixelWidth,
        canonicalRotationDegrees: canonical.rotationDegrees,
        sourcePolygon: roiPlan.normalizedMaskPolygon,
      });

      const timings = processedRoi.timings;
      console.info(
        timings
          ? `[ROI süre] parmak=${roiPlan.className}, kırpma=${cropMs}ms, kayıt=${saveMs}ms, çözme=${timings.decodeMs}ms, maske=${timings.maskMs}ms, iyileştirme=${timings.enhancementMs}ms, jpeg=${timings.encodeMs}ms, seg_yazma=${timings.writeMs}ms, kalite=${timings.qualityMs}ms, teknik_yazma=${timings.technicalWriteMs}ms, toplam=${Date.now() - fingerStartedAt}ms`
          : `[ROI süre] parmak=${roiPlan.className}, kırpma=${cropMs}ms, kayıt=${saveMs}ms, segmentasyon=başarısız, toplam=${Date.now() - fingerStartedAt}ms`
      );

      // Dosya ve analiz tamponları hazır olduğunda native ara görüntüleri serbest bırakırız.
      if (canonical.image !== cropped) {
        canonical.image.dispose();
      }
      cropped.dispose();

      return {
        ...roiPlan,
        imageUri: savedImageUri,
        canonicalImageUri,
        segmentedImageUri: processedRoi.segmentedImageUri,
        maskImageUri: processedRoi.maskImageUri,
        binaryImageUri: processedRoi.binaryImageUri,
        openedBinaryImageUri: processedRoi.openedBinaryImageUri,
        minutiaeOverlayImageUri: processedRoi.minutiaeOverlayImageUri,
        enhancedImageUri: processedRoi.enhancedImageUri,
        orientationImageUri: processedRoi.orientationImageUri,
        minutiaeImageUri: processedRoi.minutiaeImageUri,
        alignedCanonicalImageUri: processedRoi.alignedCanonicalImageUri,
        canonicalSource: sourceImage ? ('flash' as const) : ('single' as const),
        minutiaeTemplate: processedRoi.minutiaeTemplate,
        sourcePixelWidth: canonical.sourcePixelWidth,
        canonicalRotationDegrees: canonical.rotationDegrees,
        homography: processedRoi.homography,
        homographySourceSize: processedRoi.homographySourceSize,
        homographyTargetSize: processedRoi.homographyTargetSize,
        ridgeScaleFactor: processedRoi.ridgeScaleFactor,
        coordinateFrame: processedRoi.coordinateFrame,
        perspectiveCorrected: processedRoi.perspectiveCorrected,
        silhouetteAxisDegrees: processedRoi.silhouetteAxisDegrees,
        canonicalResidualDegrees: processedRoi.canonicalResidualDegrees,
        silhouetteCorrectionDegrees: processedRoi.silhouetteCorrectionDegrees,
        quality: processedRoi.quality,
      };
    }
  );

  console.info(
    `[ROI süre] parmak_sayısı=${fingerRois.length}, toplam=${Date.now() - processingStartedAt}ms`
  );

  // Paralel işçiler farklı zamanda bitse de kayıt metadata'sı her zaman kullanıcı sırasını korur.
  return sortFingerRois(fingerRois);
}

// Verilen işleri sabit sayıda işçiyle çalıştırıp sonuçların giriş sırasını korur.
async function mapWithConcurrency<Input, Output>(
  items: Input[],
  concurrency: number,
  worker: (item: Input) => Promise<Output>
) {
  const results = new Array<Output>(items.length);
  let nextIndex = 0;

  // Her işçi sıradaki boş indeksi alır; JavaScript eşzamanlı kalırken native işler bekleme süresini paylaşır.
  async function runWorker() {
    while (nextIndex < items.length) {
      const currentIndex = nextIndex;
      nextIndex += 1;
      results[currentIndex] = await worker(items[currentIndex]);
    }
  }

  const workerCount = Math.min(Math.max(concurrency, 1), items.length);
  await Promise.all(Array.from({ length: workerCount }, () => runWorker()));
  return results;
}

// Segmentasyon başarısız olursa normal ROI kaydını bozmadan boş sonuç döndürür.
async function tryProcessFingerRoi({
  roiImageUri,
  sampleId,
  className,
  sourceWidth,
  canonicalRotationDegrees,
  sourcePolygon,
}: {
  roiImageUri: string;
  sampleId: string;
  className: FingerRoi['className'];
  sourceWidth: number;
  canonicalRotationDegrees: number;
  sourcePolygon?: { x: number; y: number }[];
}) {
  try {
    const alignedCanonicalImageUri =
      await createAlignedCanonicalFingerRoiImageUri(sampleId, className);
    const outputImageUri = await createSegmentedFingerRoiImageUri(sampleId, className);
    const outputMaskImageUri = await createMaskFingerRoiImageUri(sampleId, className);
    const binaryImageUri = await createBinaryFingerRoiImageUri(sampleId, className);
    const openedBinaryImageUri =
      await createOpenedBinaryFingerRoiImageUri(sampleId, className);
    const minutiaeOverlayImageUri =
      await createMinutiaeOverlayFingerRoiImageUri(sampleId, className);
    const enhancedImageUri = await createEnhancedFingerRoiImageUri(sampleId, className);
    const orientationImageUri = await createOrientationFingerRoiImageUri(
      sampleId,
      className
    );
    const minutiaeImageUri = await createMinutiaeFingerRoiImageUri(
      sampleId,
      className
    );
    let result = await segmentFingerRoiImage({
      roiImageUri,
      outputImageUri,
      outputMaskImageUri,
      sourceWidth,
      sourcePolygon,
    });

    if (!result) {
      return {
        quality: createSegmentationFailureQuality(),
        perspectiveCorrected: false,
      };
    }

    // OBB ilk yönü verir; silüet bundan belirgin sapıyorsa ROI tamponunu sınırlı açıyla ikinci kez hizalarız.
    const initialResidualDegrees = normalizeAxisRotation(
      result.silhouetteAxisDegrees - 90
    );
    const silhouetteCorrectionDegrees =
      Math.abs(initialResidualDegrees) >= 5 &&
      Math.abs(initialResidualDegrees) <= 18
        ? -initialResidualDegrees
        : 0;
    if (silhouetteCorrectionDegrees !== 0) {
      const correctedBuffers = rotateRgbaAndMask(
        result.pixels,
        result.mask,
        result.width,
        result.height,
        silhouetteCorrectionDegrees
      );
      await saveRgbaImage({
        pixels: correctedBuffers.pixels,
        width: correctedBuffers.width,
        height: correctedBuffers.height,
        outputImageUri: alignedCanonicalImageUri,
        quality: 95,
      });
      const correctedResult = await segmentFingerRoiImage({
        roiImageUri,
        outputImageUri,
        sourceWidth,
        preparedImage: correctedBuffers,
      });
      if (correctedResult) {
        const originalHomography = result.homography;
        const originalHomographySourceSize = result.homographySourceSize;
        const originalHomographyTargetSize = result.homographyTargetSize;
        const originalPerspectiveCorrection = result.perspectiveCorrected;
        result = {
          ...correctedResult,
          homography: originalHomography,
          homographySourceSize: originalHomographySourceSize,
          homographyTargetSize: originalHomographyTargetSize,
          perspectiveCorrected: originalPerspectiveCorrection,
          timings: {
            decodeMs: result.timings.decodeMs + correctedResult.timings.decodeMs,
            maskMs: result.timings.maskMs + correctedResult.timings.maskMs,
            enhancementMs:
              result.timings.enhancementMs + correctedResult.timings.enhancementMs,
            encodeMs: result.timings.encodeMs + correctedResult.timings.encodeMs,
            writeMs: result.timings.writeMs + correctedResult.timings.writeMs,
          },
        };
      }
      console.info(
        `[ROI hizalama] silüet_düzeltmesi=${silhouetteCorrectionDegrees.toFixed(1)}°, önceki_sapma=${initialResidualDegrees.toFixed(1)}°, yeni_sapma=${normalizeAxisRotation(result.silhouetteAxisDegrees - 90).toFixed(1)}°`
      );
    } else {
      // Ek düzeltme gerekmese de kanonik görüntüyü aynı fiziksel bölgenin hizalı debug çıktısı olarak saklarız.
      await saveRgbaImage({
        pixels: result.pixels,
        width: result.width,
        height: result.height,
        outputImageUri: alignedCanonicalImageUri,
        quality: 95,
      });
    }

    console.info(
      `[ROI homografi] parmak=${className}, durum=${result.perspectiveCorrected ? 'uygulandı' : 'yedek'}`
    );
    await saveMaskPng({
      mask: result.mask,
      width: result.width,
      height: result.height,
      outputImageUri: outputMaskImageUri,
    });

    const qualityStartedAt = Date.now();
    let analysis = analyzeFingerprintQualityDetailed({
      ...result,
      sourcePixelWidth: sourceWidth,
      fingerClass: className,
    });
    const measuredRidgePeriod = analysis.quality.ridgeMedianPeriodPixels;
    const ridgeScaleFactor = calculateRidgeScaleFactor(measuredRidgePeriod);
    if (Math.abs(ridgeScaleFactor - 1) >= 0.08) {
      const scaled = rescaleRgbaAndMask({
        pixels: result.pixels,
        mask: result.mask,
        width: result.width,
        height: result.height,
        scaleFactor: ridgeScaleFactor,
      });
      result = {
        ...result,
        pixels: scaled.pixels,
        mask: scaled.mask,
        width: scaled.width,
        height: scaled.height,
        ridgeScaleFactor: scaled.scaleFactor,
      };
      await saveRgbaImage({
        pixels: result.pixels,
        width: result.width,
        height: result.height,
        outputImageUri: alignedCanonicalImageUri,
        quality: 95,
      });
      if (result.imageUri !== roiImageUri) {
        await saveRgbaImage({
          pixels: createEnhancedSegmentedPixels(
            result.pixels,
            result.mask,
            result.width,
            result.height
          ),
          width: result.width,
          height: result.height,
          outputImageUri: result.imageUri,
          quality: 90,
        });
      }
      await saveMaskPng({
        mask: result.mask,
        width: result.width,
        height: result.height,
        outputImageUri: outputMaskImageUri,
      });
      analysis = analyzeFingerprintQualityDetailed({
        ...result,
        sourcePixelWidth: sourceWidth,
        fingerClass: className,
      });
      console.info(
        `[Ridge ölçek] parmak=${className}, önce=${measuredRidgePeriod.toFixed(1)}px, sonra=${analysis.quality.ridgeMedianPeriodPixels.toFixed(1)}px, katsayı=${ridgeScaleFactor.toFixed(2)}`
      );
    }
    const quality = analysis.quality;
    const qualityMs = Date.now() - qualityStartedAt;
    let savedEnhancedImageUri: string | undefined;
    let savedOrientationImageUri: string | undefined;
    let savedMinutiaeImageUri: string | undefined;
    let savedBinaryImageUri: string | undefined;
    let savedOpenedBinaryImageUri: string | undefined;
    let savedMinutiaeOverlayImageUri: string | undefined;
    let minutiaeTemplate: FingerRoi['minutiaeTemplate'];
    let minutiaeVisualization: Uint8Array | undefined;
    let binaryVisualization: Uint8Array | undefined;
    let openedBinaryVisualization: Uint8Array | undefined;
    let minutiaeOverlayVisualization: Uint8Array | undefined;

    // Teknik Nokta görüntüsü kalite yetersiz olsa bile üretilir; yalnızca template kabulü aşağıdaki ayrı kapıda yapılır.
    if (
      analysis.enhancedPixels &&
      analysis.minutiaeSupportMask &&
      analysis.minutiaeOrientationMask &&
      quality.ridgeMedianPeriodPixels > 0
    ) {
      const minutiaeStartedAt = Date.now();
      const distalMask = createDistalMinutiaeMask(
        result.mask,
        result.width,
        result.height
      );
      const distalSupportMask = intersectMasks(
        analysis.minutiaeSupportMask,
        distalMask
      );
      const distalOrientationMask = intersectMasks(
        analysis.minutiaeOrientationMask,
        distalMask
      );
      const distalOrientationFieldMask = analysis.minutiaeOrientationFieldMask
        ? intersectMasks(analysis.minutiaeOrientationFieldMask, distalMask)
        : undefined;
      const minutiae = extractFingerprintMinutiae({
        pixels: analysis.enhancedPixels,
        mask: distalMask,
        candidateMask: distalSupportMask,
        orientationMask: distalOrientationMask,
        orientationAngles: analysis.minutiaeOrientationAngles,
        orientationFieldMask: distalOrientationFieldMask,
        width: result.width,
        height: result.height,
        ridgePeriodPixels: quality.ridgeMedianPeriodPixels,
        minimumSupportConfidence: MINUTIAE_SUPPORT_CONFIDENCE_FLOOR,
      });
      minutiaeVisualization = minutiae.visualizationPixels;
      binaryVisualization = minutiae.binaryVisualizationPixels;
      openedBinaryVisualization = minutiae.openedBinaryVisualizationPixels;
      minutiaeOverlayVisualization =
        minutiae.skeletonOverlayVisualizationPixels;
      const minutiaeAssessment = assessMinutiaeTemplate(quality, minutiae);
      quality.minutiaeStatus = minutiaeAssessment.status;
      quality.minutiaeRejectionReason = minutiaeAssessment.reason;
      quality.minutiaeCandidateCount = minutiae.template.minutiae.length;
      quality.minutiaeCrossingNumberCandidateCount =
        minutiae.crossingNumberCandidateCount;
      quality.minutiaeBranchValidatedCandidateCount =
        minutiae.branchValidatedCandidateCount;
      quality.minutiaeSuppressionCandidateCount =
        minutiae.suppressionCandidateCount;
      quality.minutiaeEndingCandidateCount = minutiae.endingCandidateCount;
      quality.minutiaeBifurcationCandidateCount =
        minutiae.bifurcationCandidateCount;
      quality.minutiaeConfidenceHistogram = minutiae.confidenceHistogram;
      quality.minutiaeThresholdPrimaryCount =
        minutiae.thresholdConsensusPrimaryCount;
      quality.minutiaeThresholdLocationStableCount =
        minutiae.thresholdConsensusLocationCount;
      quality.minutiaeThresholdTypeStableCount =
        minutiae.thresholdConsensusTypeCount;
      quality.minutiaeThresholdLoopCandidateCount =
        minutiae.thresholdLoopCandidateCount;
      quality.minutiaeThresholdLoopStableCount =
        minutiae.thresholdLoopStableCount;
      quality.minutiaeThresholdLoopRemovedCount =
        minutiae.thresholdLoopRemovedCount;
      quality.minutiaeThresholdLoopRemovedPixelCount =
        minutiae.thresholdLoopRemovedPixelCount;
      quality.minutiaeOrientationBridgeRemovedPixelCount =
        minutiae.orientationBridgeRemovedPixelCount;
      quality.minutiaeSearchableAreaRatio = Math.round(
        minutiae.searchableAreaRatio * 100
      );
      quality.minutiaeLargestRegionRatio = Math.round(
        minutiae.largestSearchableRegionRatio * 100
      );
      quality.minutiaeTopology = minutiae.topologyDiagnostics;
      if (minutiaeAssessment.status === 'sufficient') {
        minutiaeTemplate = {
          ...minutiae.template,
          textureDescriptor: createRidgeTextureDescriptor({
            pixels: analysis.enhancedPixels,
            mask: result.mask,
            width: result.width,
            height: result.height,
          }),
          ridgeScaleFactor: result.ridgeScaleFactor,
          coordinateFrame: result.perspectiveCorrected
            ? 'homography-canonical'
            : 'source-roi',
        };
      }
      const endingCount = minutiae.template.minutiae.filter(
        (item) => item.type === 'ending'
      ).length;
      const bifurcationCount =
        minutiae.template.minutiae.length - endingCount;
      console.info(
        `[Minutiae] parmak=${className}, durum=${minutiaeAssessment.status}, ret=${minutiaeAssessment.reason ?? 'yok'}, toplam=${minutiae.template.minutiae.length}, son=${endingCount}, çatallanma=${bifurcationCount}, crossing=${minutiae.crossingNumberCandidateCount}, çatallanma_akış=${minutiae.bifurcationCrossingNumberCandidateCount}/${minutiae.bifurcationBranchValidatedCount}/${minutiae.bifurcationRingValidatedCount}/${minutiae.bifurcationCandidateCount}/${minutiae.bifurcationMicroCycleValidatedCount}/${minutiae.bifurcationStabilityValidatedCount}/${minutiae.bifurcationSuppressionCount}, eşik_kararlılığı=${minutiae.thresholdConsensusPrimaryCount}/${minutiae.thresholdConsensusLocationCount}/${minutiae.thresholdConsensusTypeCount}, eşik_halkası=${minutiae.thresholdLoopCandidateCount}/${minutiae.thresholdLoopStableCount}/${minutiae.thresholdLoopRemovedCount}/${minutiae.thresholdLoopRemovedPixelCount}px, branch=${minutiae.branchValidatedCandidateCount}, suppression=${minutiae.suppressionCandidateCount}, ham_aday=${minutiae.rawCandidateCount}, iskelet=${minutiae.skeletonPixelCount}, mikro_delik=${minutiae.filledHolePixelCount}, ikili_gürültü=${minutiae.binaryNoisePixelCount}, opening=${minutiae.morphologicalOpeningStatus}, opening_px=${minutiae.morphologicalOpeningBeforePixelCount}->${minutiae.morphologicalOpeningAfterPixelCount}, opening_bileşen=${minutiae.morphologicalOpeningBeforeComponentCount}->${minutiae.morphologicalOpeningAfterComponentCount}, yön_köprüsü=${minutiae.orientationBridgeRemovedPixelCount}, küçük_bileşen=${minutiae.removedComponentPixelCount}, budanan=${minutiae.prunedPixelCount}, bağlanan=${minutiae.bridgedPixelCount}, destek=${Math.round(minutiae.supportCoverage * 100)}%, aranabilir=${Math.round(minutiae.searchableAreaRatio * 100)}%, büyük_bölge=${Math.round(minutiae.largestSearchableRegionRatio * 100)}%, ridge_oranı=${Math.round(minutiae.binaryRidgeRatio * 100)}%, inceltme=${minutiae.thinningIterations}, conf=${minutiae.confidenceHistogram.join('/')}, süre=${Date.now() - minutiaeStartedAt}ms`
      );
      const topology = minutiae.topologyDiagnostics;
      console.info(
        `[Minutiae topoloji] parmak=${className}, ikili=${formatTopologyStage(topology.binary)}, opening=${formatTopologyStage(topology.opened ?? topology.binary)}, yön_temiz=${formatTopologyStage(topology.orientationCleaned ?? topology.opened ?? topology.binary)}, inceltilmiş=${formatTopologyStage(topology.thinned)}, köprü=${formatTopologyStage(topology.bridged)}, bileşen=${formatTopologyStage(topology.componentFiltered)}, budama=${formatTopologyStage(topology.pruned)}, eşik_halkası=${formatTopologyStage(topology.thresholdLoopCleaned ?? topology.pruned)}`
      );
    } else {
      quality.minutiaeStatus = 'insufficient';
      quality.minutiaeRejectionReason =
        quality.biometricStatus === 'sufficient'
          ? 'search-area'
          : 'capture-quality';
    }

    const technicalWriteStartedAt = Date.now();
    const technicalImageWrites: Promise<unknown>[] = [];

    // Yön haritası ridge kanıtı çıkmasa da orientation hesabını doğrulayabilmek için saklanır.
    if (analysis.orientationPixels) {
      technicalImageWrites.push(
        saveRidgeEnhancedImage({
          pixels: analysis.orientationPixels,
          width: result.width,
          height: result.height,
          outputImageUri: orientationImageUri,
        }).then((savedUri) => {
          savedOrientationImageUri = savedUri;
        })
      );
    }

    // Enhancement yalnızca ridge kanıtı bulunan alan varsa yazılır; boş teknik çıktı galeriye eklenmez.
    if (analysis.enhancedPixels && analysis.enhancementSupportedAreaRatio > 0.02) {
      technicalImageWrites.push(
        saveRidgeEnhancedImage({
          pixels: analysis.enhancedPixels,
          width: result.width,
          height: result.height,
          outputImageUri: enhancedImageUri,
        }).then((savedUri) => {
          savedEnhancedImageUri = savedUri;
        })
      );
    }

    // Minutiae doğrulama görselini diğer teknik çıktılarla aynı anda kaydederiz.
    if (minutiaeVisualization) {
      technicalImageWrites.push(
        saveRidgeEnhancedImage({
          pixels: minutiaeVisualization,
          width: result.width,
          height: result.height,
          outputImageUri: minutiaeImageUri,
        }).then((savedUri) => {
          savedMinutiaeImageUri = savedUri;
        })
      );
    }
    if (binaryVisualization) {
      technicalImageWrites.push(
        saveMaskPng({
          mask: binaryVisualization,
          width: result.width,
          height: result.height,
          outputImageUri: binaryImageUri,
        }).then((savedUri) => {
          savedBinaryImageUri = savedUri;
        })
      );
    }
    if (openedBinaryVisualization) {
      technicalImageWrites.push(
        saveMaskPng({
          mask: openedBinaryVisualization,
          width: result.width,
          height: result.height,
          outputImageUri: openedBinaryImageUri,
        }).then((savedUri) => {
          savedOpenedBinaryImageUri = savedUri;
        })
      );
    }
    if (minutiaeOverlayVisualization) {
      technicalImageWrites.push(
        saveRgbaPng({
          pixels: minutiaeOverlayVisualization,
          width: result.width,
          height: result.height,
          outputImageUri: minutiaeOverlayImageUri,
        }).then((savedUri) => {
          savedMinutiaeOverlayImageUri = savedUri;
        })
      );
    }
    // Native dosya yazımları birbirini beklemeden ilerler; Promise.all iki JPEG çıktısının tamamlandığını garanti eder.
    await Promise.all(technicalImageWrites);
    const technicalWriteMs = Date.now() - technicalWriteStartedAt;
    const canonicalResidualDegrees = normalizeAxisRotation(
      result.silhouetteAxisDegrees - 90
    );

    // Saha kalibrasyonunda farklı çekimleri karşılaştırmak için yalnızca terminale kısa kalite özeti yazar.
    const biometricRejections =
      quality.biometricRejectionReasons?.join('+') || 'yok';
    console.info(
      `[ROI kalite] parmak=${className}, çekim=${quality.captureStatus}, biyometri=${quality.biometricStatus}, biyometri_ret=${biometricRejections}, genel=${quality.globalScore}, çözünürlük=${quality.sourceResolutionScore}, ridge=${quality.ridgePeriodicity}, frekans=${quality.ridgeFrequencyConsistency}, yön=${quality.orientationCoherence}, güvenilir_yön=${quality.orientationReliableBlockRatio}%, ince_yön=${quality.orientationDetailBlockSize ?? 0}px/${quality.orientationDetailVerifiedRatio ?? 0}%, yön_düzeltme=${quality.orientationMedianCorrectionDegrees.toFixed(1)}°, obb_dönüş=${canonicalRotationDegrees.toFixed(1)}°, silüet_ekseni=${result.silhouetteAxisDegrees.toFixed(1)}°, dikey_sapma=${canonicalResidualDegrees.toFixed(1)}°, ölçek=${quality.ridgeAnalysisBlockSize ?? 0}px/${quality.ridgeAnalyzedScaleCount ?? 1}, iyileştirme=${quality.ridgeEnhancementGainPercent.toFixed(1)}%, gabor_alanı=${quality.ridgeEnhancementSupportedAreaRatio}%, yön_adayı=${quality.ridgeOrientationCandidateRatio ?? 0}%, kanıt=${quality.ridgeValidBlockCount}/${quality.ridgeCandidateBlockCount} (${quality.ridgeValidBlockRatio}%, gereken=${quality.biometricRequiredValidBlockCount ?? 0}/22%), iç_blok=${quality.ridgeInteriorBlockCount}/${quality.ridgeOrientationBlockCount}, periyot=${quality.ridgeMedianPeriodPixels.toFixed(1)}px [${quality.ridgePeriodHistogram}], frekans_ret=${quality.ridgeRejectionSummary}`
    );

    return {
      segmentedImageUri: result.imageUri,
      maskImageUri: outputMaskImageUri,
      binaryImageUri: savedBinaryImageUri,
      openedBinaryImageUri: savedOpenedBinaryImageUri,
      minutiaeOverlayImageUri: savedMinutiaeOverlayImageUri,
      alignedCanonicalImageUri,
      enhancedImageUri: savedEnhancedImageUri,
      orientationImageUri: savedOrientationImageUri,
      minutiaeImageUri: savedMinutiaeImageUri,
      minutiaeTemplate,
      homography: result.homography,
      homographySourceSize: result.homographySourceSize,
      homographyTargetSize: result.homographyTargetSize,
      ridgeScaleFactor: result.ridgeScaleFactor,
      perspectiveCorrected: result.perspectiveCorrected,
      coordinateFrame: result.perspectiveCorrected
        ? ('homography-canonical' as const)
        : ('source-roi' as const),
      silhouetteAxisDegrees: result.silhouetteAxisDegrees,
      canonicalResidualDegrees,
      silhouetteCorrectionDegrees,
      quality,
      timings: {
        ...result.timings,
        qualityMs,
        technicalWriteMs,
      },
    };
  } catch (error) {
    console.warn('Parmak ROI segmentasyonu uygulanamadı.', error);
    return {
      quality: createSegmentationFailureQuality(),
      perspectiveCorrected: false,
    };
  }
}

function intersectMasks(left: Uint8Array, right: Uint8Array) {
  const result = new Uint8Array(left.length);
  for (let index = 0; index < result.length; index += 1) {
    if (left[index] && right[index]) result[index] = 1;
  }
  return result;
}

function formatTopologyStage(
  stage: MinutiaeExtractionResult['topologyDiagnostics']['binary']
) {
  return `${stage.componentCount}b/${stage.pixelCount}p/${stage.endingPixelCount}s/${stage.bifurcationPixelCount}ç`;
}

// Teknik aday listesini çekim kalitesi, kesintisiz arama alanı ve aday sayısıyla template kabulünden ayırır.
function assessMinutiaeTemplate(
  quality: FingerprintQuality,
  minutiae: MinutiaeExtractionResult
): {
  status: 'sufficient' | 'insufficient';
  reason?: MinutiaeRejectionReason;
} {
  if (quality.biometricStatus !== 'sufficient') {
    return { status: 'insufficient', reason: 'capture-quality' };
  }
  if (
    minutiae.searchableAreaRatio < MIN_SEARCHABLE_AREA_RATIO ||
    minutiae.largestSearchableRegionRatio <
      MIN_LARGEST_SEARCHABLE_REGION_RATIO
  ) {
    return { status: 'insufficient', reason: 'search-area' };
  }
  if (minutiae.template.minutiae.length < MIN_TEMPLATE_MINUTIAE_COUNT) {
    return { status: 'insufficient', reason: 'candidate-count' };
  }
  if (minutiae.template.minutiae.length > MAX_TEMPLATE_MINUTIAE_COUNT) {
    return { status: 'insufficient', reason: 'candidate-overflow' };
  }
  return { status: 'sufficient' };
}

// OBB'nin uzun kenarını dikey eksene getirip yalnızca gerçek açılı kutu boyutunu saklar.
async function createCanonicalFingerRoi(cropped: NitroImage, roiPlan: FingerRoi) {
  const geometry = getCanonicalGeometry(roiPlan.maskPolygon);
  if (!geometry) {
    return {
      image: cropped,
      sourcePixelWidth: Math.min(cropped.width, cropped.height),
      rotationDegrees: 0,
    };
  }

  const rotated = await cropped.rotateAsync(geometry.rotationDegrees, false);
  let canonical: NitroImage;
  try {
    const targetWidth = Math.min(
      Math.max(Math.round(geometry.shortEdge), 1),
      rotated.width
    );
    const targetHeight = Math.min(
      Math.max(Math.round(geometry.longEdge), 1),
      rotated.height
    );
    const left = Math.max(0, Math.round((rotated.width - targetWidth) / 2));
    const top = Math.max(0, Math.round((rotated.height - targetHeight) / 2));
    canonical = await rotated.cropAsync(
      left,
      top,
      Math.min(left + targetWidth, rotated.width),
      Math.min(top + targetHeight, rotated.height)
    );
  } finally {
    rotated.dispose();
  }

  return {
    image: canonical,
    sourcePixelWidth: geometry.shortEdge,
    rotationDegrees: geometry.rotationDegrees,
  };
}

// Kanonik çifti analiz çözünürlüğünde RGBA tamponlarına taşıyıp karşılaştırma ve gerçek poz birleşimini birlikte üretir.
// Nitro Image'in beklediği yerel dosya yolundan file URI önekini kaldırır.
function toNativeFilePath(uri: string) {
  return decodeURIComponent(uri.replace(/^file:\/\//, ''));
}

// Nitro Image'in geçici dosya yolunu Expo FileSystem'in kullanacağı URI'ye dönüştürür.
function toFileUri(path: string) {
  return path.startsWith('file://') ? path : `file://${path}`;
}

// Tek tespitin OBB köşelerinden yalnızca model kutusunu çevreleyen ROI kırpma alanını hesaplar.
function createFingerRoiPlan(
  detection: DetectedObbBox,
  imageSize: ImageSize
): FingerRoi | null {
  const axisBox = getNormalizedAxisAlignedBox(detection);
  const boxWidth = axisBox.right - axisBox.left;
  const boxHeight = axisBox.bottom - axisBox.top;

  // Geçersiz veya sıfıra yakın kutuları kayda sokmadan atlarız.
  if (boxWidth <= 0 || boxHeight <= 0) {
    return null;
  }

  const normalizedCrop = normalizeCrop({
    x: axisBox.left,
    y: axisBox.top,
    width: boxWidth,
    height: boxHeight,
  });
  const pixelCrop = toPixelCrop(normalizedCrop, imageSize);

  // Dosya olarak anlamlı olmayacak kadar küçük ROI planlarını erken eleriz.
  if (pixelCrop.width < MIN_ROI_PIXEL_SIZE || pixelCrop.height < MIN_ROI_PIXEL_SIZE) {
    return null;
  }

  return {
    id: `roi-${detection.className}`,
    detectionId: detection.id,
    className: detection.className,
    confidence: detection.confidence,
    maskPolygon: createCropRelativePolygon(detection, pixelCrop, imageSize),
    normalizedMaskPolygon: createNormalizedCropRelativePolygon(detection, pixelCrop, imageSize),
    crop: {
      ...pixelCrop,
      normalized: normalizedCrop,
    },
  };
}

// OBB köşelerini kırpılmış ROI dosyasının kendi piksel koordinat sistemine taşır.
function createCropRelativePolygon(
  detection: DetectedObbBox,
  crop: { originX: number; originY: number; width: number; height: number },
  imageSize: ImageSize
) {
  return detection.points.map((point) => ({
    x: point.x * imageSize.width - crop.originX,
    y: point.y * imageSize.height - crop.originY,
  }));
}

// OBB köşelerini kırpılmış ROI dosyasının 0-1 oranlı koordinat sistemine taşır.
function createNormalizedCropRelativePolygon(
  detection: DetectedObbBox,
  crop: { originX: number; originY: number; width: number; height: number },
  imageSize: ImageSize
) {
  return detection.points.map((point) => ({
    x: clamp01((point.x * imageSize.width - crop.originX) / Math.max(crop.width, 1)),
    y: clamp01((point.y * imageSize.height - crop.originY) / Math.max(crop.height, 1)),
  }));
}

// Normalize OBB noktalarından kırpma hesabına uygun eksene hizalı çevre kutusunu çıkarır.
function getNormalizedAxisAlignedBox(detection: DetectedObbBox) {
  const xValues = detection.points.map((point) => point.x);
  const yValues = detection.points.map((point) => point.y);

  return {
    left: Math.min(...xValues),
    top: Math.min(...yValues),
    right: Math.max(...xValues),
    bottom: Math.max(...yValues),
  };
}

// ROI planını görüntü sınırları içinde kalacak şekilde 0-1 aralığına sıkıştırır.
function normalizeCrop(crop: { x: number; y: number; width: number; height: number }) {
  const left = clamp01(crop.x);
  const top = clamp01(crop.y);
  const right = clamp01(crop.x + crop.width);
  const bottom = clamp01(crop.y + crop.height);

  return {
    x: left,
    y: top,
    width: Math.max(0, right - left),
    height: Math.max(0, bottom - top),
  };
}

// Normalize kırpma alanını ImageManipulator'ın beklediği piksel koordinatlarına dönüştürür.
function toPixelCrop(
  crop: { x: number; y: number; width: number; height: number },
  imageSize: ImageSize
) {
  const originX = Math.round(crop.x * imageSize.width);
  const originY = Math.round(crop.y * imageSize.height);
  const right = Math.round((crop.x + crop.width) * imageSize.width);
  const bottom = Math.round((crop.y + crop.height) * imageSize.height);

  return {
    originX,
    originY,
    width: Math.max(0, Math.min(imageSize.width - originX, right - originX)),
    height: Math.max(0, Math.min(imageSize.height - originY, bottom - originY)),
  };
}

// Değeri görüntü oranı olarak güvenli 0-1 aralığında tutar.
function clamp01(value: number) {
  return Math.min(Math.max(value, 0), 1);
}
