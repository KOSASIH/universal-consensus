(* formal_verification/consensus.v *)
Require Import Coq.Lists.List.
Require Import Coq.Numbers.Natural.

Section Consensus.
  Variable nodes : list node.
  Variable messages : list message.

  Inductive consensus : Prop :=
  | consensus_intro : forall (n : node) (m : message),
      In n nodes ->
      In m messages ->
      consensus.
End Consensus.
