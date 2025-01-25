import { useState, useEffect } from 'react'
import './App.css'

// @ts-ignore
import _ from 'lodash'
import * as tf from '@tensorflow/tfjs'

// take word corpus
const corpus: any = [
  'The cat sat on the porch',
  'The dog ran in the park',
  'The bird sang in the tree',
  'The bat screeched in the night'
]

const dict: any = {}

// index with vector
corpus.map((line: any) => line.split(' ')).map((words: any) => {
  words.map((word: any) => {
    if(!dict[word.toLowerCase()] && dict[word.toLowerCase()] != 0){
      dict[word.toLowerCase()] = Object.entries(dict).length
    }
  })
})

const corpus_ids: any = corpus.map((line: any) => line.split(' ')).map((words: any) => {
  return words.map((word: any) => {
    return dict[word.toLowerCase()]
  })
})

const contexts: any = []
const targets: any = []

const vocab_size: any = Object.entries(dict).length + 1
const embedding_size: any = 10
const window_size: any = 2


for(let i = 0; i < corpus_ids.length; i ++ ) {
  for(let j = 0; j < corpus_ids[i].length - window_size; j++){
    contexts.push(corpus_ids[i].filter((el: any, k: any) => {
      if(k != (window_size+j)){
        return corpus_ids[i].filter((els: any, j: any) =>{
          if(j != j + 1 + window_size + 1){
            return [el + els]  
          }
        })
      }
    }))
    targets.push(corpus_ids[i])
  }
}

const X: any = tf.tensor(contexts)
const y: any = tf.tensor(targets)

function App() {
  const [similarity, setSimilarity] = useState<any>(null)
  const [wordOne, setWordOne] = useState<any>(null)
  const [wordTwo, setWordTwo] = useState<any>(null)
  const [loading, setLoading] = useState<any>(true)
  const [model, setModel] = useState<any>(null)

  useEffect(() => {
    setTimeout(async () => {

    if(localStorage.getItem('tensorflowjs_models/my-model-1/info')){

      const loadedModel = await tf.loadLayersModel('localstorage://my-model-1');

      // const saveResults = await model.save('localstorage://my-model-1');
      setModel(loadedModel)
      setLoading(false)
    } else {
        console.log('fitting model')
        setLoading(true)

        const model = tf.sequential()

        model.add(tf.layers.embedding({inputDim: vocab_size, outputDim: embedding_size, inputLength: (2* window_size +1) }))
        model.add(tf.layers.flatten())
        // @ts-ignore
        model.add(tf.layers.dense({units: targets[0].length, activation: 'softmax'}))

        // # Compile the model
        model.compile({optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy']});
        await model.fit(X, y, {epochs: 100})
        await model.save('localstorage://my-model-1')
        setLoading(false)

      }
    }, 0)

  }, [model, loading])

  // Helper functions
  const dotProduct = (a: any, b: any) => {
    let product = 0;
    for (let i = 0; i < a.length; i++) {
      product += a[i] * b[i];
    }
    return product;
  };

  const magnitude = (vector: any) => {
    let sum = 0;
    for (let value of vector) {
      sum += value * value;
    }
    return Math.sqrt(sum);
  };

  const cosineSimilarity = (a: any, b: any) => {
    return dotProduct(a, b) / (magnitude(a) * magnitude(b));
  };

  const getSimiliarity = async () => {
    const loadedModel = model

    const idsOne: any = wordOne.split(' ').map((word: any) => {
      if(dict[word]){
        return dict[word]
      } else {
        return null
      }
    })

    console.log(idsOne)

    const idsTwo: any = wordTwo.split(' ').map((word: any) => {
      if(dict[word]){
        return dict[word]
      } else {
        return null
      }
    })

    if(idsOne.length <= 5){
      for(let i = 0; idsOne.length < 5; i++){
        idsOne.push(null)
      }
    }

    if(idsTwo.length <= 5){
      for(let i = 0; idsTwo.length < 5; i++){
        idsTwo.push(null)
      }
    }

    console.log(idsOne)
    console.log(idsTwo)
    const input = tf.tensor2d(idsOne, [1, 5])
    const input2 = tf.tensor2d(idsTwo, [1, 5])

    const similarity = cosineSimilarity(Array.from(loadedModel.predict(input).dataSync()), Array.from(loadedModel.predict(input2).dataSync()));
    setSimilarity(`${(similarity * 100).toFixed(2)}% similarity`)
  }

  return (
    <>
      <h1>Vector Similiarity Example</h1>
      {loading ? 
        <>
          loading...
        </>
        :
        <>
          <span><b>search one</b>: {wordOne}</span>
          <span><b>search two</b>: {wordTwo}</span>
          <p>{similarity}</p>
          <div className="card">
            <input onChange={(evt: any) => setWordOne(evt.target.value)}></input>
            <input onChange={(evt: any) => setWordTwo(evt.target.value)}></input>
            <button onClick={() => getSimiliarity()}>
              get similiarity
            </button>
          </div>
          <p>trained corpus</p>
          <div className="card">
              [
              <br/>

              'The cat sat on the porch',
              <br/>
              'The dog ran in the park',
              <br/>

              'The bird sang in the tree',
              <br/>

              'The bat screeched in the night'
              <br/>

              ]
          </div>
      </>
    }
    </>
  )
}

export default App
