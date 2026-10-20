//! Port of silk/float/corrMatrix_FLP.c.

/**********************************************************************
 * Correlation matrix computations for LS estimate.
 **********************************************************************/

use crate::silk::macros::Ix;
use crate::silk::float::energy_FLP::silk_energy_FLP;
use crate::silk::float::inner_product_FLP::silk_inner_product_FLP;

/* Calculates correlation vector X'*t */
pub fn silk_corrVector_FLP(
    x: &[f32],      /* I    x vector [L+order-1] used to create X       */
    t: &[f32],      /* I    Target vector [L]                           */
    L: i32,         /* I    Length of vecors                            */
    Order: i32,     /* I    Max lag for correlation                     */
    Xt: &mut [f32], /* O    X'*t correlation vector [order]             */
) {
    /* ptr1 points to first sample of column 0 of X: X[:,0]; x[Order - 1 - lag] for column lag */
    for lag in 0..Order {
        /* Calculate X[:,lag]'*t */
        *Xt.at_mut(lag as usize) = silk_inner_product_FLP(x.rg((Order - 1 - lag) as usize..), t, L) as f32;
    }
}

/* Calculates correlation matrix X'*X */
pub fn silk_corrMatrix_FLP(
    x: &[f32],       /* I    x vector [ L+order-1 ] used to create X     */
    L: i32,          /* I    Length of vectors                           */
    Order: i32,      /* I    Max lag for correlation                     */
    XX: &mut [f32],  /* O    X'*X correlation matrix [order x order]     */
) {
    // ptr1 = &x[Order - 1]: first sample of column 0 of X
    let p1 = (Order - 1) as isize;
    let at = |i: isize| x.at(i as usize);
    let (L_, Order_) = (L as isize, Order as usize);
    let mp = |row: usize, col: usize| row * Order_ + col;

    let mut energy: f64 = silk_energy_FLP(x.rg(p1 as usize..), L); /* X[:,0]'*X[:,0] */
    *XX.at_mut(mp(0, 0)) = energy as f32;
    for j in 1..Order_ {
        let jj = j as isize;
        /* Calculate X[:,j]'*X[:,j] */
        energy += (at(p1 - jj) * at(p1 - jj) - at(p1 + L_ - jj) * at(p1 + L_ - jj)) as f64;
        *XX.at_mut(mp(j, j)) = energy as f32;
    }

    // ptr2 = &x[Order - 2]: first sample of column 1 of X; decremented for each lag
    for lag in 1..Order_ {
        let p2 = p1 - lag as isize;
        /* Calculate X[:,0]'*X[:,lag] */
        energy = silk_inner_product_FLP(x.rg(p1 as usize..), x.rg(p2 as usize..), L);
        *XX.at_mut(mp(lag, 0)) = energy as f32;
        *XX.at_mut(mp(0, lag)) = energy as f32;
        /* Calculate X[:,j]'*X[:,j + lag] */
        for j in 1..(Order_ - lag) {
            let jj = j as isize;
            energy += (at(p1 - jj) * at(p2 - jj) - at(p1 + L_ - jj) * at(p2 + L_ - jj)) as f64;
            *XX.at_mut(mp(lag + j, j)) = energy as f32;
            *XX.at_mut(mp(j, lag + j)) = energy as f32;
        }
    }
}
