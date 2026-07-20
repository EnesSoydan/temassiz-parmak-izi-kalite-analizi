import * as FileSystem from 'expo-file-system/legacy';

import type { CaptureSample } from '@/types/biometrics';

const ROOT_DIR = `${FileSystem.documentDirectory ?? ''}fingerprint-captures/`;
const RAW_DIR = `${ROOT_DIR}raw/`;
const ROI_DIR = `${ROOT_DIR}roi/`;
const INDEX_FILE = `${ROOT_DIR}captures.json`;

export async function ensureCaptureStorage() {
  await ensureDirectory(ROOT_DIR);
  await ensureDirectory(RAW_DIR);
  await ensureDirectory(ROI_DIR);
}

export async function saveRawImage(sourceUri: string, sampleId: string) {
  await ensureCaptureStorage();
  const targetUri = `${RAW_DIR}${sampleId}.jpg`;
  await FileSystem.copyAsync({ from: sourceUri, to: targetUri });
  return targetUri;
}

export async function saveRoiImage(sourceUri: string, sampleId: string) {
  await ensureCaptureStorage();
  const targetUri = `${ROI_DIR}${sampleId}.jpg`;
  await FileSystem.copyAsync({ from: sourceUri, to: targetUri });
  return targetUri;
}

export async function loadCaptureSamples(): Promise<CaptureSample[]> {
  await ensureCaptureStorage();
  const info = await FileSystem.getInfoAsync(INDEX_FILE);

  if (!info.exists) {
    return [];
  }

  const text = await FileSystem.readAsStringAsync(INDEX_FILE);
  return JSON.parse(text) as CaptureSample[];
}

export async function appendCaptureSample(sample: CaptureSample) {
  const samples = await loadCaptureSamples();
  const nextSamples = [sample, ...samples];
  await FileSystem.writeAsStringAsync(INDEX_FILE, JSON.stringify(nextSamples, null, 2));
  return nextSamples;
}

export async function deleteCaptureSample(sampleId: string) {
  const samples = await loadCaptureSamples();
  const sample = samples.find((item) => item.id === sampleId);
  const nextSamples = samples.filter((item) => item.id !== sampleId);

  if (sample?.rawImageUri) {
    await deleteFileIfExists(sample.rawImageUri);
  }

  if (sample?.roiImageUri) {
    await deleteFileIfExists(sample.roiImageUri);
  }

  await FileSystem.writeAsStringAsync(INDEX_FILE, JSON.stringify(nextSamples, null, 2));
  return nextSamples;
}

async function ensureDirectory(uri: string) {
  const info = await FileSystem.getInfoAsync(uri);

  if (!info.exists) {
    await FileSystem.makeDirectoryAsync(uri, { intermediates: true });
  }
}

async function deleteFileIfExists(uri: string) {
  const info = await FileSystem.getInfoAsync(uri);

  if (info.exists) {
    await FileSystem.deleteAsync(uri, { idempotent: true });
  }
}
