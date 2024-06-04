expressions_equiv_even = [
        '(equals (mod x_ 2) 0)', # x % 2 == 0
        '(equals (mod (multiply x_ 2) 4) 0)', # (x * 2) % 4 == 0
        '(equals (mod (multiply x_ 3) 6) 0)', # (x * 3) % 6 == 0
        '(equals (mod (multiply x_ 4) 8) 0)', # (x * 4) % 8 == 0
        '(equals (mod (divide (multiply x_ 2) 2) 2) 0)', 
        '(equals (mod (divide (multiply x_ 3) 3) 2) 0)', # multiply x_ by 3, then divide by 3, then mod by 2 and check if 0
        '(equals (mod (divide (multiply x_ 4) 4) 2) 0)', 
        '(equals (mod (divide (multiply x_ 5) 5) 2) 0)', 
        '(equals (mod (divide (multiply x_ 6) 6) 2) 0)', 
        '(equals (mod (divide (multiply x_ 7) 7) 2) 0)', 
        '(equals (mod (divide (multiply x_ 8) 8) 2) 0)', 
        '(equals (mod (divide (multiply x_ 9) 9) 2) 0)', 
        ]
