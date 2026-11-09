import os

def create_isl_svg_assets(output_dir="static/signs"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Complete 50 ISL Descriptions (Alphabets, Numbers, Words, Gestures)
    isl_descriptions = {
        # Alphabets (A-Z)
        "A": ("Both hands: Left palm flat facing up, Right index finger touches Left thumb tip.", "2 Hands"),
        "B": ("Both hands: Left palm flat, Right index finger touches Left index finger tip.", "2 Hands"),
        "C": ("Right hand: Curved into a 'C' arc shape facing left.", "1 Hand"),
        "D": ("Both hands: Left index finger held straight up; Right hand forms arc touching tip and base of Left index.", "2 Hands"),
        "E": ("Both hands: Left palm flat, Right index finger touches Left middle finger tip.", "2 Hands"),
        "F": ("Both hands: Left index & middle fingers held horizontal; Right index & middle cross over them forming 'F'.", "2 Hands"),
        "G": ("Both hands: Both hands in fists with thumbs resting over index, stacked vertically.", "2 Hands"),
        "H": ("Both hands: Left palm flat facing right; Right hand sweeps horizontal across left palm.", "2 Hands"),
        "I": ("Both hands: Left palm flat, Right index finger touches Left ring finger tip.", "2 Hands"),
        "J": ("Both hands: Left palm flat, Right index traces downward hook across left palm.", "2 Hands"),
        "K": ("Both hands: Left index pointed up; Right index and thumb form a V touching left index joint.", "2 Hands"),
        "L": ("Right hand: Thumb and index finger extended at 90 degrees forming an 'L' shape.", "1 Hand"),
        "M": ("Both hands: Left open palm; Right hand puts 3 fingers (index, middle, ring) down on left palm.", "2 Hands"),
        "N": ("Both hands: Left open palm; Right hand puts 2 fingers (index, middle) down on left palm.", "2 Hands"),
        "O": ("Both hands: Left palm flat, Right index finger touches Left pinky finger tip.", "2 Hands"),
        "P": ("Both hands: Left index finger held upright; Right index and thumb form circle touching tip of Left index.", "2 Hands"),
        "Q": ("Both hands: Left hand makes circle; Right index finger tucks under circle forming hook.", "2 Hands"),
        "R": ("Both hands: Left palm open; Right index finger curled and hooked over left palm.", "2 Hands"),
        "S": ("Both hands: Left pinky finger hooked with Right pinky finger.", "2 Hands"),
        "T": ("Both hands: Left index finger held horizontal; Right index finger touches bottom of left index forming 'T'.", "2 Hands"),
        "U": ("Both hands: Left palm flat facing right, Right index finger traces cup-shaped 'U' on palm.", "2 Hands"),
        "V": ("Right hand: Index and middle fingers extended apart forming a 'V' peace sign.", "1 Hand"),
        "W": ("Both hands: Interlocking fingers of both hands upright forming 'W' peaks.", "2 Hands"),
        "X": ("Both hands: Both index fingers crossed over each other forming an 'X'.", "2 Hands"),
        "Y": ("Both hands: Left index finger extended upright; Right V-sign placed on top of Left index.", "2 Hands"),
        "Z": ("Both hands: Left palm held vertical facing side; Right index finger touches left palm diagonally.", "2 Hands"),
        
        # Numbers (0-9)
        "0": ("Right hand: Forms an 'O' ring with all fingers touching thumb.", "1 Hand"),
        "1": ("Right hand: Index finger extended straight up.", "1 Hand"),
        "2": ("Right hand: Index and middle fingers extended straight up.", "1 Hand"),
        "3": ("Right hand: Thumb, index, and middle fingers extended.", "1 Hand"),
        "4": ("Right hand: 4 fingers extended, thumb folded across palm.", "1 Hand"),
        "5": ("Right hand: Open palm with all 5 fingers spread out.", "1 Hand"),
        "6": ("Both hands: Left hand shows 5 fingers, Right hand points 1 finger.", "2 Hands"),
        "7": ("Both hands: Left hand shows 5 fingers, Right hand shows 2 fingers.", "2 Hands"),
        "8": ("Both hands: Left hand shows 5 fingers, Right hand shows 3 fingers.", "2 Hands"),
        "9": ("Both hands: Left hand shows 5 fingers, Right hand shows 4 fingers.", "2 Hands"),
        
        # Key Words
        "HELLO": ("Right hand open palm raised to temple in a polite wave/salute.", "1 Hand"),
        "NAMASTE": ("Both palms joined together in front of chest in traditional greeting.", "2 Hands"),
        "PLEASE": ("Right hand flat palm rubbed gently over chest in clockwise circle.", "1 Hand"),
        "THANK YOU": ("Right fingertips touch chin then move forward towards camera with open palm.", "1 Hand"),
        "YES": ("Right fist nodding up and down like a head nodding.", "1 Hand"),
        "NO": ("Right index, middle finger and thumb snapping together like a beak.", "1 Hand"),
        "HELP": ("Left flat palm facing up; Right thumbs-up placed firmly on left palm.", "2 Hands"),
        "GOOD": ("Right hand thumbs up gesture held in front of chest.", "1 Hand"),
        "BAD": ("Right hand thumbs down gesture held in front of chest.", "1 Hand"),
        "WATER": ("Right hand forms 'W' (index, middle, ring up) tapping chin twice.", "1 Hand"),
        "FOOD": ("Right fingers pinched together tapping lips twice.", "1 Hand"),
        "FRIEND": ("Both index fingers hooked together interlocking both hands.", "2 Hands"),
        "I LOVE YOU": ("Right hand: Thumb, Index, and Pinky fingers extended upright (middle & ring folded).", "1 Hand"),

        # Gestures
        "SPACE": ("Both hands: Palms open facing inward, sweep outward horizontally to create space.", "Gesture"),
        "BACKSPACE": ("Right hand: Fist with thumb pointing left, sweep backward horizontally.", "Gesture")
    }
    
    for letter, (desc, hand_type) in isl_descriptions.items():
        svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 320" width="100%" height="100%">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1e293b"/>
      <stop offset="100%" stop-color="#0f172a"/>
    </linearGradient>
    <linearGradient id="handGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8"/>
      <stop offset="100%" stop-color="#0284c7"/>
    </linearGradient>
  </defs>
  
  <rect width="320" height="320" rx="20" fill="url(#bgGrad)" stroke="#334155" stroke-width="3"/>
  
  <rect x="22" y="20" width="80" height="28" rx="8" fill="#1e3a8a" stroke="#60a5fa" stroke-width="1.5"/>
  <text x="62" y="39" font-family="'Segoe UI', Roboto, sans-serif" font-size="12" font-weight="700" fill="#93c5fd" text-anchor="middle">ISL SIGN</text>

  <rect x="215" y="20" width="85" height="28" rx="8" fill="#312e81" stroke="#818cf8" stroke-width="1.5"/>
  <text x="257" y="39" font-family="'Segoe UI', Roboto, sans-serif" font-size="11" font-weight="600" fill="#c7d2fe" text-anchor="middle">{hand_type}</text>

  <circle cx="160" cy="115" r="54" fill="#1e293b" stroke="#38bdf8" stroke-width="3"/>
  <text x="160" y="{128 if len(letter) > 4 else 138}" font-family="'Segoe UI', Roboto, sans-serif" font-size="{28 if len(letter) > 5 else (38 if len(letter) > 2 else 64)}" font-weight="900" fill="#38bdf8" text-anchor="middle">{letter}</text>

  <g transform="translate(110, 185) scale(0.7)">
    <path d="M40 80 C 30 70, 20 50, 35 30 C 45 15, 60 25, 65 40 L 65 15 C 65 5, 75 5, 80 15 L 80 40 L 85 10 C 90 0, 100 0, 105 10 L 105 45 L 115 20 C 120 10, 130 15, 130 25 L 125 70 C 120 95, 95 110, 75 110 C 50 110, 45 95, 40 80 Z" fill="url(#handGrad)" opacity="0.9"/>
  </g>

  <rect x="18" y="222" width="284" height="82" rx="12" fill="#0b1329" stroke="#1e293b" stroke-width="1.5"/>
  <text x="160" y="242" font-family="'Segoe UI', Roboto, sans-serif" font-size="12" font-weight="700" fill="#f8fafc" text-anchor="middle">ISL Hand Gesture Guide</text>
  <foreignObject x="24" y="248" width="272" height="52">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-family:'Segoe UI',sans-serif; font-size:11px; color:#cbd5e1; text-align:center; line-height:1.35; padding: 2px;">
      {desc}
    </div>
  </foreignObject>
</svg>"""
        file_path = os.path.join(output_dir, f"{letter}.svg")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
            
    print(f"Generated {len(isl_descriptions)} ISL sign SVG assets in '{output_dir}'.")

if __name__ == "__main__":
    create_isl_svg_assets()
