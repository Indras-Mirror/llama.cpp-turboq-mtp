You are continuing work on RotorQuant integration in a llama.cpp fork. Read this handoff FIRST by running:

  cat /home/mal/AI/llama.cpp-mtp-fixes/HANDOFF.md

Then execute these steps in order:

1. cd /home/mal/AI/llama.cpp-mtp-fixes && git checkout feature/rotorquant
2. Join the relay room:
   relay_room_join room-rotorquant-collab-1778360638274
3. If you're the lead instance: relay_rename q-lead
   If you're the peer instance: relay_rename q-peer

=== CURRENT STATE ===
ALL RotorQuant code compiles clean on sm_89 (4090). Zero build errors.
First test CRASHED with: GGML_ASSERT(ggml_can_mul_mat(k, q)) failed
Crash is in llama-graph.cpp pass-through logic for planar/iso types.
Fix is simple — see HANDOFF.md §6.

=== IMMEDIATE TASK ===
1. Fix the llama-graph.cpp crash (remove planar/iso from k_is_tbq/v_is_tbq)
2. Rebuild and test with: -ctk planar3_0 -ctv planar3_0 at 4096 context
3. If works, scale to 262K context
4. Verify TBQ4 still works (regression test with -ctk tbq4_0 -ctv tbq4_0)
5. If all good, commit, push, notify Issue #3, update README

=== TASK SPLIT (if peer connected) ===
q-lead: Fix crash, test planar3_0, benchmark
q-peer: Regression test TBQ4, audit other modified files, update COLLAB_ROTORQUANT.md
