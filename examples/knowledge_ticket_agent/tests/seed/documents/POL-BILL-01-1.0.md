# Fictional purchase credit eligibility
Synthetic: true; data_origin: authored_fiction

## eligibility | First purchase eligibility
A trusted verified identity is required. For a first purchase, request age must be 0 through 14 UTC calendar days inclusive. The amount must be positive and at most 5000 minor DEMO_CREDIT units, and the invoice must not already be credited. Amounts are integers, not real currency. Only simulated credit guidance is permitted.

## manual | High amount review
Otherwise eligible first-purchase requests from 5001 through 10000 minor DEMO_CREDIT units inclusive require manual_review. This is a review route, not an approval or an executed credit. No high-amount request may be automatically resolved.
