use neural_network::{NeuralNetwork, Layer, TestData};
use rand::Rng;
use std::{fs, iter};

fn digit_to_layer(input: u8) -> Result<impl Iterator<Item = f64>, String> {
    if !(0..=9).contains(&input) {
        return Err(format!("Input value: {} must be between 0 and 9", input));
    }

    Ok((0..=9).map(move |x| if x == input {1.0} else {0.0}))
}

fn layer_to_digit(input: &Layer) -> Result<u8, &str> {
    if input.len() != 10 {
        return Err("Input length must be 10 for a 1:1 mapping");
    }

    input
        .inspect()
        .max_by(|(_, lhs), (_, rhs)| {
            /* println!("{}, {}", *lhs, *rhs);  */
            f64::partial_cmp(lhs, rhs).expect("partial_cmp should not encounter NaNs or Infs")
        })
        .ok_or("Failed to find max value in layer, check for NaNs")?
        .0
        .try_into()
        .map_err(|_| "Failed to convert index from usize to u8")
}

fn main() {
    let mut network = {
        NeuralNetwork::new(&[28 * 28, 30, 7, 10], |_, _, _| {
            rand::thread_rng().gen_range(-1.0..1.0)
        })
        .unwrap()
    };

    let mut training_data = {
        let image_byte_data = fs::read("training/train-images.idx3-ubyte").unwrap();
        let image_iter = image_byte_data.split_at(16).1.chunks(28 * 28);

        let labels_data = fs::read("training/train-labels.idx1-ubyte").unwrap();
        let labels_slice = labels_data.split_at(8).1;

        let mut data = Vec::with_capacity(labels_slice.len());

        for (image, label) in iter::zip(image_iter, labels_slice) {
            data.push(TestData::new(
                image.iter().map(|input| f64::from(*input) / 255.0), // between 0.0 and 1.0
                digit_to_layer(*label).unwrap(),
                *label,
            ))
        }

        data.into_boxed_slice()
    };

    let mut test_data = {
        let image_byte_data = fs::read("training/t10k-images.idx3-ubyte").unwrap();
        let image_iter = image_byte_data.split_at(16).1.chunks(28 * 28);

        let labels_data = fs::read("training/t10k-labels.idx1-ubyte").unwrap();
        let labels_slice = labels_data.split_at(8).1;

        let mut data = Vec::with_capacity(labels_slice.len());

        for (image, label) in iter::zip(image_iter, labels_slice) {
            data.push(neural_network::TestData::new(
                image.iter().map(|input| f64::from(*input) / 255.0), // between 0.0 and 1.0
                digit_to_layer(*label).unwrap(),
                *label,
            ))
        }

        data.into_boxed_slice()
    };

    loop {
        network.train(&mut training_data, 20, 1.0);
        
        let test_output = network.test(&mut test_data, |input| {layer_to_digit(input).unwrap()}); 
        println!{"{}", test_output};
    }
}