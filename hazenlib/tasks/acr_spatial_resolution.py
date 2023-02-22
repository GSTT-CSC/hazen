"""
ACR Spatial Resolution (MTF)

https://www.acraccreditation.org/-/media/acraccreditation/documents/mri/largephantomguidance.pdf

Calculates the effective resolution (MTF50) for slice 1 for the ACR phantom. This is done in accordance with the
methodology described in Section 3 of the following paper:



WARNING: The phantom must be slanted for valid results to be produced. This test is not within the scope of ACR guidance.

This script calculates the bar length difference in accordance with the ACR Guidance. Line profiles are drawn
vertically through the left and right wedges. The right wedge's line profile is shifted and wrapped round before being
subtracted from the left wedge's line profile, e.g.:

Right line profile: [1, 2, 3, 4, 5]
Right line profile wrapped round by 1: [2, 3, 4, 5, 1]

This wrapping process, from hereon referred to as circular shifting, is then used for subtractions.

The shift used to produce the minimum difference between the circularly shifted right line profile and the static left
one is used to determine the bar length difference, which is twice the slice position displacement.
The results are also visualised.

Created by Yassine Azma
yassine.azma@rmh.nhs.uk

22/02/2023
"""