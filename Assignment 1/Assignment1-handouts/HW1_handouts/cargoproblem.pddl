(define (problem transport_cargo) 
 (:domain cargo_world) 
(:objects C1 C2  - cargo
 P1 P2  - plane
 SFO JFK  - airport
 ) 
(:init (cargo_at C1 SFO)(cargo_at C2 JFK)(plane_at P1 SFO)(plane_at P2 JFK)) 
(:goal (and (cargo_at C1 JFK) (cargo_at C2 SFO))) 
) 
