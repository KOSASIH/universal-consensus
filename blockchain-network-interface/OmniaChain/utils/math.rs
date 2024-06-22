// Import necessary libraries and dependencies
use num::{BigInt, BigUint};
use rand::Rng;

// Define the Math struct
pub struct Math {
    // Random number generator
    rng: rand::ThreadRng,
}

impl Math {
    // Create a new Math instance
    pub fn new() -> Self {
        Math {
            rng: rand::thread_rng(),
        }
    }

    // Generate a random prime number
    pub fn random_prime(&self, bits: usize) -> BigUint {
        let mut prime = BigUint::new(vec![0u8; bits / 8]);
        while !is_prime(&prime) {
            prime = BigUint::new(vec![0u8; bits / 8]);
            self.rng.fill_bytes(prime.as_mut_bytes());
        }
        prime
    }

    // Check if a number is prime
    pub fn is_prime(&self, number: &BigUint) -> bool {
        if number <= &BigUint::from(1u8) {
            return false;
        }
        if number <= &BigUint::from(3u8) {
            return true;
        }
        if number % &BigUint::from(2u8) == BigUint::from(0u8) {
            return false;
        }
        let mut i = BigUint::from(3u8);
        while i * i <= number {
            if number % &i == BigUint::from(0u8) {
                return false;
            }
            i += BigUint::from(2u8);
        }
        true
    }

    // Calculate the greatest common divisor (GCD) of two numbers
    pub fn gcd(&self, a: &BigUint, b: &BigUint) -> BigUint {
        let mut a = a.clone();
        let mut b = b.clone();
        while b != BigUint::from(0u8) {
            let temp = b.clone();
            b = a % &b;
            a = temp;
        }
        a
    }

    // Calculate the modular inverse of a number
    pub fn mod_inverse(&self, a:&BigUint, n: &BigUint) -> Option<BigUint> {
        let g = self.gcd(a, n);
        if g != BigUint::from(1u8) {
            return None;
        }
        let mut x = BigUint::from(0u8);
        let mut y = BigUint::from(1u8);
        let mut a = a.clone();
        let mut b = n.clone();
        while a > BigUint::from(0u8) {
            let q = &b / &a;
            let temp = a.clone();
            a = b.clone() % a;
            b = temp;
            let temp = x.clone();
            x = y.clone() - q * x;
            y = temp;
        }
        if y < BigUint::from(0u8) {
            y += n.clone();
        }
        Some(y)
    }
}

// Define the is_prime function
fn is_prime(number: &BigUint) -> bool {
    // Check if the number is prime using the Miller-Rabin primality test
    let mut d = number.clone() - BigUint::from(1u8);
    let mut r = 0u32;
    while d % BigUint::from(2u8) == BigUint::from(0u8) {
        r += 1;
        d /= BigUint::from(2u8);
    }
    let mut a = BigUint::from(2u8);
    while a < number {
        let mut x = BigUint::from(1u8);
        let mut y = a.clone();
        for _ in 0..r {
            y = (y * y) % number;
        }
        if y == BigUint::from(1u8) || y == *number {
            a += BigUint::from(1u8);
            continue;
        }
        for _ in 0..(r - 1) {
            y = (y * y) % number;
            if y == *number {
                return false;
            }
        }
        x = y;
        if x == BigUint::from(1u8) || x == *number {
            return false;
        }
        a += BigUint::from(1u8);
    }
    true
}
