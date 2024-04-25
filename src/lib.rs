//! This library is for generating, training and using a neural network.
//! A network can be fully generated with the `NeuralNetwork` struct and associated functions,
//! and implementation details regarding the layers and neurons has been hidden.
//!
//! A list of `TestData<T>` can be used to both input the relevant neurons to the NN and to store
//! the actual training result (and a more readable label for convenience)

#![feature(array_windows)]
#![feature(lint_reasons)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

pub mod matrix;
pub mod vector;

use core::{
    fmt::Debug,
    iter,
    ops::{self, Index},
};
use rand::prelude::*;

extern crate alloc;

struct WindowsMut<'x, const N: usize, T> {
    slice: &'x mut [T],
    forward_index: usize,
    rev_index: usize,
}

impl<'x, const N: usize, T> WindowsMut<'x, N, T> {
    fn new(slice: &'x mut [T]) -> Self {
        let length = slice.len();
        Self {
            slice,
            forward_index: 0,
            rev_index: length,
        }
    }

    fn next(&mut self) -> Option<&mut [T; N]> {
        let Self {
            slice,
            forward_index,
            rev_index: _,
        } = self;
        let ans = slice[*forward_index..].first_chunk_mut();
        *forward_index += 1;
        ans
    }

    fn next_back(&mut self) -> Option<&mut [T; N]> {
        let Self {
            slice,
            forward_index: _,
            rev_index,
        } = self;
        let ans = slice[..*rev_index].last_chunk_mut();
        *rev_index -= 1;
        ans
    }
}

/// ReLU (Leaky Linear Rectified Unit) used to map negative numbers to much smaller negatives.
pub fn relu(val: f64) -> f64 {
    if val > 0.0 {
        val
    } else {
        val * 0.1
    }
}

/// `tall_sigmoid` function maps all input to between (-1.0, 1.0) in an s-curve
pub fn tall_sigmoid(val: f64) -> f64 {
    2.0 / (1.0 / val.exp() + 1.0) - 1.0
}

/// `sigmoid` function maps all input to between (0, 1) in an s-curve
pub fn sigmoid(val: f64) -> f64 {
    1.0 / (1.0 / val.exp() + 1.0)
}

/// A convenience dot product fn which takes two IntoIterators and perform a sum-product
/// ```
///     # use neural_network::dot_product;
///     assert!(dot_product([1,2,3], [4,5,6]) == (1 * 4) + (2 * 5) + (3 * 6));
/// ```
pub fn dot_product<B, A: ops::Mul<B, Output = T>, T: iter::Sum>(
    lhs: impl IntoIterator<Item = A>,
    rhs: impl IntoIterator<Item = B>,
) -> T
/* where
    L: IntoIterator<Item = I>,
    R: IntoIterator<Item = I>,
    I: ops::Mul<Output = T>,

    T: iter::Sum, */
{
    iter::zip(lhs, rhs).map(|(x, y)| x * y).sum()
}

/// Sum-Square of the difference betweek two iterators. The magnitude of the difference between two vectors
pub fn cost<L, R, I, T>(lhs: L, rhs: R) -> T
where
    L: IntoIterator<Item = I>,
    R: IntoIterator<Item = I>,
    I: ops::Sub<Output = T> + Copy,
    T: ops::Mul<Output = T>,

    T: iter::Sum,
{
    iter::zip(lhs, rhs)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<T>()
}

// the `cost` function applied to an output `Layer` and a `TestData<T>`
pub fn cost_test<T: PartialEq>(test: &TestData<T>, output: &Layer) -> f64 {
    cost(
        test.inspect_answer().map(|(_, activation)| activation),
        output.iter().map(|neuron| neuron.activation()),
    )
    .sqrt()
}

/// A Neuron in the NeuralNetwork. Contains an activation and a list of weights corresponding to the previous layer.
/// Not directly constructible, will likely be hidden as it is never interacted with.
#[derive(Clone)]
struct Neuron {
    activation: f64,
    weights: Box<[f64]>, // len of this vec = number of previous neurons + 1 (for itself, the bias)
}

#[derive(Clone)]
pub struct Layer {
    neurons: Box<[Neuron]>, // len of this vec = number of neurons in layer
}

impl Neuron {
    fn new_from_parts(activation: f64, weights: Box<[f64]>) -> Self {
        Self {
            activation,
            weights,
        }
    }

    fn calculate(&mut self, input: &Layer, function: impl Fn(f64) -> f64) -> Result<(), String> {
        if input.len() != self.num_connections() {
            return Err(format!(
                "Incompatible Neuron weights {} and input layer {}",
                self.num_connections(),
                input.len()
            ));
        }

        let unrect = dot_product(
            input.inspect_with_unit(),
            self.weights.iter(),
        );
        self.set_activation(function(unrect / (self.weights.len() as f64)));

        Ok(())
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = &f64> {
        self.weights.iter()
    }

    fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = &mut f64> {
        self.weights.iter_mut()
    }

    pub fn activation(&self) -> f64 {
        self.activation
    }

    fn set_activation(&mut self, value: f64) {
        self.activation = value;
    }

    fn incr_activation(&mut self, increment: f64) {
        self.activation += increment;
    }

    fn num_connections(&self) -> usize {
        self.weights.len() - 1
    }
}

#[derive(Default)]
struct LayerBuilder {
    count: Option<usize>,
    activations: Option<Vec<f64>>,
    weights: Option<Vec<Vec<f64>>>,
}

impl LayerBuilder {
    fn add_neurons(mut self, activations: impl IntoIterator<Item = f64>) -> Self {
        self.activations = Some(activations.into_iter().collect());
        self
    }

    fn add_by_count(mut self, count: usize) -> Self {
        self.count = Some(count);
        self
    }

    fn add_weights(
        mut self,
        weights: impl IntoIterator<Item = impl IntoIterator<Item = f64>>,
    ) -> Self {
        self.weights = Some(
            weights
                .into_iter()
                .map(|iter| iter.into_iter().collect() )
                .collect(),
        );
        self
    }

    fn build(self) -> Result<Layer, &'static str> {
        let activations = match self.activations {
            Some(mut activations)  => {
                if let Some(count) = self.count {
                    activations.resize(count, 0.0);
                }
                Ok(activations)
            },
            None => match self.count {
                Some(count) => {Ok(iter::repeat(0.0).take(count).collect())}
                None => Err("One of a neuron count or a list of activations must be provided. Activations take precedence but count can be used to truncate or extend by empty neurons")
            },
        }?;

        let weights: Vec<Box<[f64]>> = match self.weights {
            Some(weights) => {
                if weights.len() != activations.len() {
                    return Err("Input weights vector different length to neurons");
                }
                let mut iter = weights.iter().map(|vec| vec.len());

                let val = iter.next().ok_or("Input weights vector is empty")?;
                for len in iter {
                    if len != val {
                        return Err("Neuron weights are not same length");
                    }
                }
                Ok::<_, &str>(weights.into_iter().map(|vec| vec.into_boxed_slice()).collect())
            }
            None => Ok((0..activations.len()).map(|_| Vec::new().into_boxed_slice()).collect()),
        }?;

        Ok(Layer {
            neurons: iter::zip(activations, weights)
                .map(|(activation, weights)| Neuron::new_from_parts(activation, weights))
                .collect(),
        })
    }
}

impl Layer {
    fn builder() -> LayerBuilder {
        Default::default()
    }

    fn calculate(
        &mut self,
        previous: &Layer,
        rectifier: impl Fn(f64) -> f64,
    ) -> Result<(), String> {
        for (index, neuron) in self.iter_mut().enumerate() {
            neuron
                .calculate(previous, &rectifier)
                .map_err(|mut err_str| {
                    err_str.push_str(&format!(" on neuron {index}"));
                    err_str
                })?;
        }
        Ok(())
    }


    /// Find the length of the Layer
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.neurons.len()
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = &Neuron> {
        self.neurons.iter()
    }

    fn iter_mut(&mut self) -> core::slice::IterMut<Neuron> {
        self.neurons.iter_mut()
    }

    fn iter_num_connections(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.iter().map(Neuron::num_connections)
    }

    fn num_connections(&self) -> usize {
        self.iter_num_connections().next().unwrap()
    }

    pub fn inspect(&self) -> impl ExactSizeIterator<Item = (usize, f64)> + '_ {
        self.neurons
            .iter()
            .map(|neuron| neuron.activation())
            .enumerate()
    }

    fn inspect_with_unit(&self) -> impl Iterator<Item = f64> + '_ {
        self.neurons
        .iter()
        .map(|neuron| neuron.activation())
        .chain(iter::once(1.0))
    }
}

impl Debug for Neuron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.activation())
    }
}

impl Debug for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.neurons)
    }
}

pub struct TestData<T> {
    input: Layer,
    answer: Layer,
    value: T,
}

impl<T> TestData<T> {
    pub fn new(
        input: impl IntoIterator<Item = f64>,
        answer: impl IntoIterator<Item = f64>,
        value: T,
    ) -> Self {
        Self {
            input: Layer::builder().add_neurons(input).build().unwrap(),
            answer: Layer::builder().add_neurons(answer).build().unwrap(),
            value,
        }
    }

    pub fn input(&self) -> &Layer {
        &self.input
    }

    fn inspect_answer(&self) -> impl ExactSizeIterator<Item = (usize, f64)> + '_ {
        self.answer.inspect()
    }
}

impl<T: Clone> TestData<T> {
    pub fn value(&self) -> T {
        self.value.clone()
    }
}

pub struct NeuralNetwork {
    layers: Box<[Layer]>,
    rectifier: Box<[Box<dyn Fn(f64) -> f64>]>,
}

pub struct TrainingOutput {
    cost: f64,
    proportion: f64,
}

impl core::fmt::Display for TrainingOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.0}: {:.2}",
            self.cost,
            self.proportion * 100.0
        )
    }
}

pub struct NeuralNetworkBuilder {

}

impl NeuralNetwork {
    pub fn new(
        layer_sizes: &[usize],
        mut weight_value: impl FnMut(usize, usize, usize) -> f64,
    ) -> Result<Self, String> {
        let mut layers = Vec::new();

        for (curr_layer, [input_len, current_len]) in layer_sizes.array_windows().enumerate() {
            let layer = Layer::builder()
                .add_by_count(*current_len)
                .add_weights(
                    (0..*current_len)
                        .map(|curr_row| {
                            (0..input_len + 1) // + 1 to account for the bias
                                .map(|input_row| weight_value(curr_layer, input_row, curr_row))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>(),
                )
                .build()?;

            layers.push(layer);
        }

        for num_connections in layers.index(0).iter_num_connections() {
            if num_connections != layers.index(0).num_connections() {
                return Err(format!(
                    "Incorrect pairing between input length and layer 0 weights: {} to {}",
                    num_connections,
                    layers.index(0).num_connections()
                ));
            }
        }

        for (layer_index, [input_layer, current_layer]) in layers.array_windows().enumerate() {
            let input_layer_len = input_layer.len();
            for (neuron_index, neuron) in current_layer.iter().enumerate() {
                if neuron.num_connections() != input_layer_len {
                    return Err(format!("Incorrect pairing between layer {} length and layer {}, neuron {} weights: {} to {}", layer_index , layer_index + 1, neuron_index, input_layer_len, neuron.num_connections()));
                }
            }
        }

        let mut rectifier: Vec<Box<dyn Fn(f64) -> f64>> = Vec::with_capacity(layers.len());

        rectifier.resize_with(layers.len() - 1, || Box::new(relu)); // Can later adjust for ReLU
        rectifier.push(Box::new(sigmoid));

        Ok(Self { layers: layers.into_boxed_slice() , rectifier: rectifier.into_boxed_slice() })
    }

    pub fn layer_sizes(&self) -> impl Iterator<Item = usize> + '_ {
        iter::once(self.input_count()).chain(self.layers.iter().map(|layer| layer.len()))
    }

    fn input_count(&self) -> usize {
        self.layers.index(0).num_connections()
    }

    pub fn calculate<'a>(&'a mut self, input: &'a Layer) -> Result<&'a Layer, String> {
        if self.input_count() != input.len() {
            return Err(format!(
                "Network input expectation ({}) is not the same as the given input ({})",
                self.input_count(),
                input.len()
            ));
        }

        // To start, the calculate needs to look at the input and the first element of the list
        let mut iter_rectifier = self.rectifier.iter();

        self.layers
            .first_mut()
            .ok_or_else(|| "Empty NN".to_string())?
            .calculate(
                input,
                iter_rectifier
                    .next()
                    .ok_or_else(|| "Empty rectifier list".to_string())?,
            )?;

        let mut layers = WindowsMut::new(&mut self.layers);
        let mut index = 0_usize;
        while let (Some([input, current]), Some(rectifier)) = (layers.next(), iter_rectifier.next())
        {
            current.calculate(input, rectifier).map_err(|mut err_str| {
                err_str.push_str(&format!(" in layer {index}"));
                err_str
            })?;
            index += 1;
        }

        self.layers
            .last()
            .ok_or_else(|| "Error finding output layer".to_owned())
    }

    pub fn back_propogate<T>(
        &mut self,
        evaluated_nn: &NeuralNetwork,
        test_data: &TestData<T>,
    ) -> Result<(), String> {
        if !self.layer_sizes().eq(evaluated_nn.layer_sizes()) {
            return Err("Shape of the self and evaluated NN are not the same".to_owned());
        }
        if self.input_count() != test_data.input().len() {
            return Err(format!("Input count for self and the evaluated NN ({}) is different to the test data provided ({})", self.input_count(), test_data.input().len()));
        }

        // Configure the target for the last layer to be the correct answer
        for (((_, answer), eval), destination) in iter::zip(
            iter::zip(
                test_data.inspect_answer(),
                evaluated_nn.layers.last().unwrap().iter(),
            ),
            self.layers.last_mut().unwrap().iter_mut(),
        ) {
            destination.set_activation(answer - eval.activation());
        }

        // Change target values first
        // [         Layer X - 1         ] [          Layer X          ]
        // [target_input] (for mut values) [target_current] (for values)
        //                                 [current_layer] (for weights)
        //
        // Window spans from  first to last values in the target network (including the output layer,
        // not including the added input layer which cannot be modified)
        //
        // adjust each neuron in the target_input by modifying based on the activation of the

        // target_input_val += constant * weight * target_current

        let mut iter_target = WindowsMut::new(&mut self.layers);
        let mut iter_eval = evaluated_nn.layers.iter();

        while let (Some([target_input_layer, target_current_layer]), Some(current_layer)) =
            (iter_target.next_back(), iter_eval.next_back())
        {
            // zero-out the activations in the input layer, before they are incremented
            for target_input_neuron in target_input_layer.iter_mut() {
                target_input_neuron.set_activation(0.0);
            }

            // At this point I now have all three layers which I need to back-propogate.
            // For each (current_neuron, target_current_neuron) pair I will zip the weights of the current layer with the neurons
            // of the target_input.

            let average_factor = 1.0 / (current_layer.len() as f64);

            for (current_neuron, target_current_neuron) in
                iter::zip(current_layer.iter(), target_current_layer.iter_mut())
            {
                for (weight, target_input_neuron) in
                    iter::zip(current_neuron.iter(), target_input_layer.iter_mut())
                {
                    target_input_neuron.incr_activation(
                        average_factor * (*weight) * target_current_neuron.activation(),
                    );
                }
            }
        }

        // Change weight values next
        // [      Layer X - 1       ] [          Layer X                          ]
        //                            [target_current] (for values and mut weights)
        // [input_layer] (for values)
        //
        //

        for (input_layer, target_current) in iter::zip(
            evaluated_nn
                .layers
                .iter()
                .rev()
                .skip(1)
                .chain(iter::once(test_data.input())),
            self
                .layers
                .iter_mut()
                .rev(),
        ) {
            for target_current_neuron in target_current.iter_mut() {
                let target_current_activation = target_current_neuron.activation();
                for (input_neuron_activation, target_weight) in iter::zip(
                    input_layer.inspect_with_unit(),
                    target_current_neuron.iter_mut(),
                ) {
                    *target_weight += input_neuron_activation * target_current_activation;
                }
            }
        }

        Ok(())
    }

    pub fn all_weights(&self) -> impl Iterator<Item = &f64> {
        self.layers
            .iter()
            .flat_map(|layer| layer.iter())
            .flat_map(|neuron| neuron.iter())
    }

    fn all_weights_mut(&mut self) -> impl Iterator<Item = &mut f64> {
        self.layers
            .iter_mut()
            .flat_map(|layer| layer.iter_mut())
            .flat_map(|neuron| neuron.iter_mut())
    }

    pub fn train<T>(&mut self, data: &mut [TestData<T>], chunk_size: usize, learning_rate: f64) {
        let inv_chunk_size = learning_rate / chunk_size as f64;       
        
        data.shuffle(&mut rand::thread_rng());
        for chunk in data.chunks(chunk_size) {
            let mut change_network = Self::new(&self.layer_sizes().collect::<Vec<_>>(), |_, _, _| 0.0).unwrap();
            for test in chunk {
                self.calculate(test.input()).unwrap();

                change_network.back_propogate(self, test).unwrap();
            }

            for (destination, source) in iter::zip(self.all_weights_mut(), change_network.all_weights()) {
                *destination += *source * inv_chunk_size;
            }
        }
    }

    pub fn test<T: PartialEq + Clone>(&mut self, data: &mut [TestData<T>], test_fn: impl Fn(&Layer) -> T) -> TrainingOutput {
        let mut correct = 0.0;
        let mut cost = 0.0;
        
        let data_len = data.len() as f64;

        for test in data {
            let output = self.calculate(test.input()).unwrap();
            
            correct += (test_fn(output) == test.value()) as u8 as f64;
            cost += cost_test(test, output);
        }

        TrainingOutput {cost, proportion: correct / data_len} 
    }
}
