# Independent L1 Relation Audit Protocol

## Purpose

This protocol evaluates relations admitted by one frozen pipeline. Candidate
relations and relations from an older pipeline may be audited diagnostically,
but they are not evidence that the current admission policy meets its target.

## Blinding and independence

1. Freeze the packet and record its SHA-256 before annotation.
2. Each annotator labels a separate copy of the complete packet.
3. Annotators must not inspect model probabilities, `model_key.json`, GPT/model
   annotations, verifier outputs, development/held-out membership, or another
   annotator's labels.
4. Adjudication happens only after both independent copies are locked.
5. GPT labels are `model_provisional`; they never satisfy the human gate.

## Judgments

- `supported`: the supplied evidence establishes the requested relation.
- `unsupported`: the supplied evidence does not establish it.
- `contradicted`: supplied evidence conflicts with it.
- `unclear`: evidence is insufficient to decide safely.

Identity requires grounded entity endpoints, compatible type, positive identity
evidence, feasible time/trajectory, and no simultaneous-distinct or stable
attribute conflict. Similar wording, pronouns, and temporal proximity are not
identity evidence by themselves.

State transition additionally requires one accepted identity track, the same
normalized attribute, different visible before/after values, and correct time
direction. Repeated or compatible state is not a transition.

## Locked evaluation

Report all three quantities for each required relation group:

- decided precision = supported / (supported + unsupported + contradicted)
- conservative precision = supported / all admitted, including unclear
- decision coverage = decided / all admitted

The formal gate requires conservative precision >= 0.90 and decision coverage
>= 0.95 for both identity and state-transition groups. Both groups must be
non-empty. Model-provisional labels always produce `acceptance_passed=false`.

Use a hidden deterministic split manifest for development and held-out item
IDs. Do not tune rules on held-out labels. The existing 45-edge packet is an
old-flow diagnostic; after the verifier is frozen, generate a new packet from
the relations actually admitted by that verifier.
