import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';

// ─────────────────────────────────────────────────────────
// Label Arrays (hardcoded to avoid parsing Python .pkl)
// ─────────────────────────────────────────────────────────

/**
 * 28 alphabet classes — matches the sklearn LabelEncoder order
 * from models/label_encoder.pkl
 */
export const ALPHABET_LABELS: string[] = [
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
  'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space',
];

/**
 * 21 dynamic phrase classes — matches models/label_map.json
 * Index → label mapping (label_map.json stores label → index)
 */
export const DYNAMIC_LABELS: string[] = [
  'hello', 'yes', 'no', 'thankyou', 'please', 'fine', 'who',
  'where', 'why', 'water', 'food', 'sleep', 'hungry', 'thirsty',
  'sick', 'go', 'wait', 'drink', 'read', 'talk', 'listen',
];

// ─────────────────────────────────────────────────────────
// Model Asset References
// ─────────────────────────────────────────────────────────

// Alphabet model: input [1, 63] → output [1, 28]
const alphabetModelJSON = require('../assets/models/alphabet/model.json');
const alphabetModelWeights: number = require('../assets/models/alphabet/group1-shard1of1.bin');

// Dynamic model: input [1, 30, 63] → output [1, 21]
const dynamicModelJSON = require('../assets/models/dynamic/model.json');
const dynamicModelWeights: number = require('../assets/models/dynamic/group1-shard1of1.bin');

// ─────────────────────────────────────────────────────────
// ModelLoader Class
// ─────────────────────────────────────────────────────────

export interface LoadedModels {
  alphabetModel: tf.LayersModel;
  dynamicModel: tf.LayersModel;
}

/**
 * Loads both TFJS LayersModels at app startup using bundleResourceIO.
 *
 * bundleResourceIO is the React Native–specific loader that reads
 * model.json + .bin shards from the Metro asset bundle.
 *
 * @returns Both loaded models ready for .predict()
 */
export async function loadModels(): Promise<LoadedModels> {
  console.log('[ModelLoader] Loading Alphabet model...');
  const alphabetModel = await tf.loadLayersModel(
    bundleResourceIO(alphabetModelJSON, alphabetModelWeights)
  );
  console.log(
    `[ModelLoader] ✅ Alphabet model loaded — ` +
    `input: ${JSON.stringify(alphabetModel.inputs[0].shape)}, ` +
    `output: ${JSON.stringify(alphabetModel.outputs[0].shape)}`
  );

  console.log('[ModelLoader] Loading Dynamic model...');
  const dynamicModel = await tf.loadLayersModel(
    bundleResourceIO(dynamicModelJSON, dynamicModelWeights)
  );
  console.log(
    `[ModelLoader] ✅ Dynamic model loaded — ` +
    `input: ${JSON.stringify(dynamicModel.inputs[0].shape)}, ` +
    `output: ${JSON.stringify(dynamicModel.outputs[0].shape)}`
  );

  return { alphabetModel, dynamicModel };
}
