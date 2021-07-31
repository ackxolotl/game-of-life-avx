#![feature(stdsimd)]

// Conway's Game of Life using AVX instructions
// Compile with `rustc +nightly -C opt-level=3 -C target-cpu=native src/main.rs`.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::time::Instant;

const STEPS: usize = 1_000_000_000;

#[repr(C)]
union M256 {
    u8s: (
        (u8, u8, u8, u8, u8, u8, u8, u8),
        (u8, u8, u8, u8, u8, u8, u8, u8),
        (u8, u8, u8, u8, u8, u8, u8, u8),
        (u8, u8, u8, u8, u8, u8, u8, u8),
    ),
    u32s: (u32, u32, u32, u32, u32, u32, u32, u32),
    m256i: __m256i,
}

impl From<&Vec<u8>> for M256 {
    fn from(v: &Vec<u8>) -> Self {
        M256 {
            u8s: (
                ((v[1]  << 4) +  v[0], (v[3]  << 4) +  v[2], (v[5]  << 4) +  v[4], (v[7]  << 4) +  v[6], (v[9]  << 4) +  v[8], (v[11] << 4) + v[10], (v[13] << 4) + v[12], (v[15] << 4) + v[14]),
                ((v[17] << 4) + v[16], (v[19] << 4) + v[18], (v[21] << 4) + v[20], (v[23] << 4) + v[22], (v[25] << 4) + v[24], (v[27] << 4) + v[26], (v[29] << 4) + v[28], (v[31] << 4) + v[30]),
                ((v[33] << 4) + v[32], (v[35] << 4) + v[34], (v[37] << 4) + v[36], (v[39] << 4) + v[38], (v[41] << 4) + v[40], (v[43] << 4) + v[42], (v[45] << 4) + v[44], (v[47] << 4) + v[46]),
                ((v[49] << 4) + v[48], (v[51] << 4) + v[50], (v[53] << 4) + v[52], (v[55] << 4) + v[54], (v[57] << 4) + v[56], (v[59] << 4) + v[58], (v[61] << 4) + v[60], (v[63] << 4) + v[62]),
            ),
        }
    }
}

impl Into<Vec<u8>> for M256 {
    fn into(self) -> Vec<u8> {
        unsafe {
            vec![
                (self.u8s.0).0 & 0xf, (self.u8s.0).0 >> 4, (self.u8s.0).1 & 0xf, (self.u8s.0).1 >> 4, (self.u8s.0).2 & 0xf, (self.u8s.0).2 >> 4, (self.u8s.0).3 & 0xf, (self.u8s.0).3 >> 4,
                (self.u8s.0).4 & 0xf, (self.u8s.0).4 >> 4, (self.u8s.0).5 & 0xf, (self.u8s.0).5 >> 4, (self.u8s.0).6 & 0xf, (self.u8s.0).6 >> 4, (self.u8s.0).7 & 0xf, (self.u8s.0).7 >> 4,
                (self.u8s.1).0 & 0xf, (self.u8s.1).0 >> 4, (self.u8s.1).1 & 0xf, (self.u8s.1).1 >> 4, (self.u8s.1).2 & 0xf, (self.u8s.1).2 >> 4, (self.u8s.1).3 & 0xf, (self.u8s.1).3 >> 4,
                (self.u8s.1).4 & 0xf, (self.u8s.1).4 >> 4, (self.u8s.1).5 & 0xf, (self.u8s.1).5 >> 4, (self.u8s.1).6 & 0xf, (self.u8s.1).6 >> 4, (self.u8s.1).7 & 0xf, (self.u8s.1).7 >> 4,
                (self.u8s.2).0 & 0xf, (self.u8s.2).0 >> 4, (self.u8s.2).1 & 0xf, (self.u8s.2).1 >> 4, (self.u8s.2).2 & 0xf, (self.u8s.2).2 >> 4, (self.u8s.2).3 & 0xf, (self.u8s.2).3 >> 4,
                (self.u8s.2).4 & 0xf, (self.u8s.2).4 >> 4, (self.u8s.2).5 & 0xf, (self.u8s.2).5 >> 4, (self.u8s.2).6 & 0xf, (self.u8s.2).6 >> 4, (self.u8s.2).7 & 0xf, (self.u8s.2).7 >> 4,
                (self.u8s.3).0 & 0xf, (self.u8s.3).0 >> 4, (self.u8s.3).1 & 0xf, (self.u8s.3).1 >> 4, (self.u8s.3).2 & 0xf, (self.u8s.3).2 >> 4, (self.u8s.3).3 & 0xf, (self.u8s.3).3 >> 4,
                (self.u8s.3).4 & 0xf, (self.u8s.3).4 >> 4, (self.u8s.3).5 & 0xf, (self.u8s.3).5 >> 4, (self.u8s.3).6 & 0xf, (self.u8s.3).6 >> 4, (self.u8s.3).7 & 0xf, (self.u8s.3).7 >> 4,
            ]
        }
    }
}

impl M256 {
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))]
    unsafe fn step(&mut self) {
        // calculate number of neighbors by rotating and adding the current
        // universe by one position into all possible directions

        let mut neighbors = _mm256_setzero_si256();

        // rotate up
        let idx = M256 { u32s: (1, 2, 3, 4, 5, 6, 7, 0)};
        let tmp = _mm256_permutevar8x32_epi32(self.m256i, idx.m256i);
        neighbors = _mm256_add_epi64(neighbors, tmp);

        // rotate up + left
        neighbors = _mm256_add_epi64(neighbors, _mm256_xor_si256(_mm256_srli_epi32(tmp, 4), _mm256_slli_epi32(tmp, 28)));

        // rotate up + right
        neighbors = _mm256_add_epi64(neighbors, _mm256_xor_si256(_mm256_slli_epi32(tmp, 4), _mm256_srli_epi32(tmp, 28)));

        // rotate left
        let tmp = self.m256i;
        neighbors = _mm256_add_epi64(neighbors, _mm256_xor_si256(_mm256_srli_epi32(tmp, 4), _mm256_slli_epi32(tmp, 28)));

        // rotate right
        neighbors = _mm256_add_epi64(neighbors, _mm256_xor_si256(_mm256_slli_epi32(tmp, 4), _mm256_srli_epi32(tmp, 28)));

        // rotate down
        let idx = M256 { u32s: (7, 0, 1, 2, 3, 4, 5, 6)};
        let tmp = _mm256_permutevar8x32_epi32(self.m256i, idx.m256i);
        neighbors = _mm256_add_epi64(neighbors, tmp);

        // rotate down + left
        neighbors = _mm256_add_epi64(neighbors, _mm256_xor_si256(_mm256_srli_epi32(tmp, 4), _mm256_slli_epi32(tmp, 28)));

        // rotate down + right
        neighbors = _mm256_add_epi64(neighbors, _mm256_xor_si256(_mm256_slli_epi32(tmp, 4), _mm256_srli_epi32(tmp, 28)));

        // universe OR neighbors
        // all future living cells equal 0x3, all other values are dead cells
        neighbors = _mm256_or_si256(neighbors, self.m256i);

        // upper nibble of every byte == 0x3?
        let upper_nibbles = _mm256_and_si256(_mm256_cmpeq_epi8 (_mm256_and_si256(neighbors, _mm256_set1_epi8(-16i8)), _mm256_set1_epi8(0x30)), _mm256_set1_epi8(0x10));

        // lower nibble of every byte == 0x3?
        let lower_nibbles = _mm256_and_si256(_mm256_cmpeq_epi8 (_mm256_and_si256(neighbors, _mm256_set1_epi8(0xf)), _mm256_set1_epi8(3)), _mm256_set1_epi8(1));

        // update universe
        self.m256i = _mm256_xor_si256(upper_nibbles, lower_nibbles);
    }

    fn print(&self) {
        unsafe {
            println!("{} {} {} {} {} {} {} {}", (self.u8s.0).0 & 0xf, (self.u8s.0).0 >> 4, (self.u8s.0).1 & 0xf, (self.u8s.0).1 >> 4, (self.u8s.0).2 & 0xf, (self.u8s.0).2 >> 4, (self.u8s.0).3 & 0xf, (self.u8s.0).3 >> 4);
            println!("{} {} {} {} {} {} {} {}", (self.u8s.0).4 & 0xf, (self.u8s.0).4 >> 4, (self.u8s.0).5 & 0xf, (self.u8s.0).5 >> 4, (self.u8s.0).6 & 0xf, (self.u8s.0).6 >> 4, (self.u8s.0).7 & 0xf, (self.u8s.0).7 >> 4);
            println!("{} {} {} {} {} {} {} {}", (self.u8s.1).0 & 0xf, (self.u8s.1).0 >> 4, (self.u8s.1).1 & 0xf, (self.u8s.1).1 >> 4, (self.u8s.1).2 & 0xf, (self.u8s.1).2 >> 4, (self.u8s.1).3 & 0xf, (self.u8s.1).3 >> 4);
            println!("{} {} {} {} {} {} {} {}", (self.u8s.1).4 & 0xf, (self.u8s.1).4 >> 4, (self.u8s.1).5 & 0xf, (self.u8s.1).5 >> 4, (self.u8s.1).6 & 0xf, (self.u8s.1).6 >> 4, (self.u8s.1).7 & 0xf, (self.u8s.1).7 >> 4);
            println!("{} {} {} {} {} {} {} {}", (self.u8s.2).0 & 0xf, (self.u8s.2).0 >> 4, (self.u8s.2).1 & 0xf, (self.u8s.2).1 >> 4, (self.u8s.2).2 & 0xf, (self.u8s.2).2 >> 4, (self.u8s.2).3 & 0xf, (self.u8s.2).3 >> 4);
            println!("{} {} {} {} {} {} {} {}", (self.u8s.2).4 & 0xf, (self.u8s.2).4 >> 4, (self.u8s.2).5 & 0xf, (self.u8s.2).5 >> 4, (self.u8s.2).6 & 0xf, (self.u8s.2).6 >> 4, (self.u8s.2).7 & 0xf, (self.u8s.2).7 >> 4);
            println!("{} {} {} {} {} {} {} {}", (self.u8s.3).0 & 0xf, (self.u8s.3).0 >> 4, (self.u8s.3).1 & 0xf, (self.u8s.3).1 >> 4, (self.u8s.3).2 & 0xf, (self.u8s.3).2 >> 4, (self.u8s.3).3 & 0xf, (self.u8s.3).3 >> 4);
            println!("{} {} {} {} {} {} {} {}", (self.u8s.3).4 & 0xf, (self.u8s.3).4 >> 4, (self.u8s.3).5 & 0xf, (self.u8s.3).5 >> 4, (self.u8s.3).6 & 0xf, (self.u8s.3).6 >> 4, (self.u8s.3).7 & 0xf, (self.u8s.3).7 >> 4);
        }

        println!();
    }
}

fn main() {
    let universe_with_glider = vec![
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 1, 0, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    ];

    let mut universe = M256::from(&universe_with_glider);

    println!(
        "Simulating {} billion steps of this universe: \n",
        STEPS / 1_000_000_000
    );

    for _ in 0..3 {
        universe.print();

        println!("       |       ");
        println!("       v       \n");

        unsafe {
            universe.step();
        }
    }

    println!("      ...      \n");

    let now = Instant::now();

    for _ in 0..STEPS {
        unsafe {
            universe.step();
        }
    }

    let elapsed = now.elapsed().as_millis() as f64 / 1000.0;

    println!(
        "{} billion steps took {:.3} seconds, that is about {} million steps per second!",
        STEPS / 1_000_000_000,
        elapsed,
        (STEPS as f64 / elapsed) as usize / 1_000_000
    );
}

#[test]
fn test_block() {
    let before = vec![
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 1,
    ];

    let after: Vec<u8> = unsafe {
        let mut universe = M256::from(&before);
        universe.step();
        universe.into()
    };

    assert_eq!(before, after);
}

#[test]
fn test_loaf() {
    let before = vec![
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 1,
        0, 1, 0, 0, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    ];

    let after: Vec<u8> = unsafe {
        let mut universe = M256::from(&before);
        universe.step();
        universe.into()
    };

    assert_eq!(before, after);
}

#[test]
fn test_blinker() {
    let one = vec![
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    ];

    let two = vec![
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    ];

    let after: Vec<u8> = unsafe {
        let mut universe = M256::from(&one);
        universe.step();
        universe.into()
    };

    assert_eq!(two, after);

    let after: Vec<u8> = unsafe {
        let mut universe = M256::from(&two);
        universe.step();
        universe.into()
    };

    assert_eq!(one, after);
}
