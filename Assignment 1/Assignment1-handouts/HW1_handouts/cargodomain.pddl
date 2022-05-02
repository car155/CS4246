(define (domain cargo_world ) 
(:requirements :strips :typing) 
(:types cargo
plane
airport
) 
(:predicates (cargo_at ?C - cargo ?A - airport) 
(plane_at ?P - plane ?A - airport) 
(in ?C - cargo ?P - plane) 
) 
(:action LOAD
:parameters ( ?C - cargo ?P - plane ?A - airport) 
:precondition (and (cargo_at ?C ?A) (plane_at ?P ?A))
:effect (and (not (cargo_at ?C ?A)) (in ?C ?P))
) 
(:action UNLOAD
:parameters ( ?C - cargo ?P - plane ?A - airport) 
:precondition (and (in ?C ?P) (plane_at ?P ?A))
:effect (and (cargo_at ?C ?A) (not (in ?C ?P)))
) 
(:action FLY
:parameters ( ?P - plane ?from - airport ?to - airport) 
:precondition (plane_at ?P ?from)
:effect (and (not (plane_at ?P ?from)) (plane_at ?P ?to))
) 
) 
