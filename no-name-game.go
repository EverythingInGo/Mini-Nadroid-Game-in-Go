// main.go
package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"golang.org/x/mobile/app"
	"golang.org/x/mobile/event/lifecycle"
	"golang.org/x/mobile/event/paint"
	"golang.org/x/mobile/event/size"
	"golang.org/x/mobile/event/touch"
	"golang.org/x/mobile/gl"
)

// ========================
// CONSTANTS & CONFIG
// ========================
const (
	ScreenWidth     = 1280
	ScreenHeight    = 720
	FOV             = math.Pi / 3
	PlayerSpeed     = 5.0
	PlayerTurnSpeed = 2.5
	BulletSpeed     = 25.0
	MaxBullets      = 32
	MaxEnemies      = 8
	PhysicsTickHz   = 60
	ArenaSize       = 50.0 // 50x50 meter arena
)

// ========================
// VECTOR3 TYPE & OPERATIONS
// ========================
type Vec3 struct {
	X, Y, Z float32
}

func (v Vec3) Add(o Vec3) Vec3 {
	return Vec3{v.X + o.X, v.Y + o.Y, v.Z + o.Z}
}
func (v Vec3) Sub(o Vec3) Vec3 {
	return Vec3{v.X - o.X, v.Y - o.Y, v.Z - o.Z}
}
func (v Vec3) Mul(s float32) Vec3 {
	return Vec3{v.X * s, v.Y * s, v.Z * s}
}
func (v Vec3) Length() float32 {
	return float32(math.Sqrt(float64(v.X*v.X + v.Y*v.Y + v.Z*v.Z)))
}
func (v Vec3) Normalize() Vec3 {
	len := v.Length()
	if len == 0 {
		return v
	}
	return Vec3{v.X / len, v.Y / len, v.Z / len}
}
func Cross(a, b Vec3) Vec3 {
	return Vec3{
		a.Y*b.Z - a.Z*b.Y,
		a.Z*b.X - a.X*b.Z,
		a.X*b.Y - a.Y*b.X,
	}
}
func Dot(a, b Vec3) float32 {
	return a.X*b.X + a.Y*b.Y + a.Z*b.Z
}

// ========================
// PLAYER STRUCT
// ========================
type Player struct {
	Position  Vec3
	Yaw       float32 // radians
	Pitch     float32 // radians
	Velocity  Vec3
	Health    int32
	Ammo      int32
	Alive     bool
	Mutex     sync.Mutex
	LastShot  time.Time
}

// ========================
// BULLET STRUCT
// ========================
type Bullet struct {
	Position Vec3
	Velocity Vec3
	Active   bool
}

// ========================
// ENEMY STRUCT
// ========================
type Enemy struct {
	Position Vec3
	Health   int32
	Alive    bool
	Mutex    sync.Mutex
}

// ========================
// GAME STATE STRUCT
// ========================
type GameState struct {
	Player  Player
	Bullets [MaxBullets]Bullet
	Enemies [MaxEnemies]Enemy
	Mutex   sync.Mutex
}

var gameState = GameState{}

// ========================
// RENDERER STRUCT
// ========================
type Renderer struct {
	glctx          gl.Context
	program        gl.Program
	vertexAttrib   gl.Attrib
	colorAttrib    gl.Attrib
	matrixUniform  gl.Uniform
	width, height  int
	projMatrix     [16]float32
	viewMatrix     [16]float32
	modelMatrix    [16]float32
}

var renderer *Renderer

// ========================
// SHADER SOURCES
// ========================
var vertexShaderSrc = `
attribute vec3 vertexPosition;
attribute vec4 vertexColor;
uniform mat4 modelViewProj;
varying vec4 fragColor;
void main() {
	gl_Position = modelViewProj * vec4(vertexPosition, 1.0);
	fragColor = vertexColor;
}
`

var fragmentShaderSrc = `
precision mediump float;
varying vec4 fragColor;
void main() {
	gl_FragColor = fragColor;
}
`

// ========================
// MATRIX UTILITIES
// ========================

func perspective(fov, aspect, near, far float32) [16]float32 {
	f := 1.0 / float32(math.Tan(float64(fov/2)))
	nf := 1 / (near - far)
	return [16]float32{
		f / aspect, 0, 0, 0,
		0, f, 0, 0,
		0, 0, (far + near) * nf, -1,
		0, 0, (2 * far * near) * nf, 0,
	}
}

func identity() [16]float32 {
	return [16]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
}

func translate(m [16]float32, v Vec3) [16]float32 {
	m[12] += v.X
	m[13] += v.Y
	m[14] += v.Z
	return m
}

func rotateY(m [16]float32, angle float32) [16]float32 {
	c := float32(math.Cos(float64(angle)))
	s := float32(math.Sin(float64(angle)))

	r := identity()
	r[0] = c
	r[2] = s
	r[8] = -s
	r[10] = c

	return multiply(m, r)
}

func multiply(a, b [16]float32) [16]float32 {
	var r [16]float32
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			r[i*4+j] = a[i*4+0]*b[0*4+j] + a[i*4+1]*b[1*4+j] + a[i*4+2]*b[2*4+j] + a[i*4+3]*b[3*4+j]
		}
	}
	return r
}

func lookAt(eye, center, up Vec3) [16]float32 {
	f := center.Sub(eye).Normalize()
	s := Cross(f, up).Normalize()
	u := Cross(s, f)

	m := identity()

	m[0] = s.X
	m[4] = s.Y
	m[8] = s.Z

	m[1] = u.X
	m[5] = u.Y
	m[9] = u.Z

	m[2] = -f.X
	m[6] = -f.Y
	m[10] = -f.Z

	m[12] = -Dot(s, eye)
	m[13] = -Dot(u, eye)
	m[14] = Dot(f, eye)

	return m
}

// ========================
// INIT RENDERER
// ========================
func NewRenderer(glctx gl.Context, width, height int) *Renderer {
	r := &Renderer{glctx: glctx, width: width, height: height}

	vertexShader := compileShader(glctx, gl.VERTEX_SHADER, vertexShaderSrc)
	fragmentShader := compileShader(glctx, gl.FRAGMENT_SHADER, fragmentShaderSrc)

	prog := glctx.CreateProgram()
	glctx.AttachShader(prog, vertexShader)
	glctx.AttachShader(prog, fragmentShader)
	glctx.LinkProgram(prog)

	var linked bool
	glctx.GetProgramiv(prog, gl.LINK_STATUS, &linked)
	if !linked {
		log.Fatalf("Could not link program")
	}

	r.program = prog
	r.vertexAttrib = glctx.GetAttribLocation(prog, "vertexPosition")
	r.colorAttrib = glctx.GetAttribLocation(prog, "vertexColor")
	r.matrixUniform = glctx.GetUniformLocation(prog, "modelViewProj")

	r.glctx.EnableVertexAttribArray(r.vertexAttrib)
	r.glctx.EnableVertexAttribArray(r.colorAttrib)

	r.projMatrix = perspective(FOV, float32(width)/float32(height), 0.1, 100)
	return r
}

func compileShader(glctx gl.Context, shaderType uint32, src string) gl.Shader {
	shader := glctx.CreateShader(shaderType)
	glctx.ShaderSource(shader, src)
	glctx.CompileShader(shader)

	var compiled bool
	glctx.GetShaderiv(shader, gl.COMPILE_STATUS, &compiled)
	if !compiled {
		log.Fatalf("Failed to compile shader: %v", glctx.GetShaderInfoLog(shader))
	}
	return shader
}

// ========================
// DRAW SIMPLE CUBE (player/enemy/bullet)
// ========================

var cubeVertices = []float32{
	// positions
	-0.5, -0.5, -0.5,
	0.5, -0.5, -0.5,
	0.5, 0.5, -0.5,
	-0.5, 0.5, -0.5,
	-0.5, -0.5, 0.5,
	0.5, -0.5, 0.5,
	0.5, 0.5, 0.5,
	-0.5, 0.5, 0.5,
}

var cubeIndices = []uint16{
	0, 1, 2, 2, 3, 0,
	1, 5, 6, 6, 2, 1,
5, 4, 7, 7, 6, 5,
4, 0, 3, 3, 7, 4,
3, 2, 6, 6, 7, 3,
4, 5, 1, 1, 0, 4,
}

var cubeColors = []float32{
1, 0, 0, 1,
0, 1, 0, 1,
0, 0, 1, 1,
1, 1, 0, 1,
1, 0, 1, 1,
0, 1, 1, 1,
0.5, 0.5, 0.5, 1,
1, 1, 1, 1,
}

func (r *Renderer) DrawCube(pos Vec3, scale float32, color [4]float32) {
glctx := r.glctx

vertices := make([]float32, 0, len(cubeVertices)*2)
colors := make([]float32, 0, len(cubeColors))

for i := 0; i < len(cubeVertices)/3; i++ {
	// vertex pos scaled and translated
	vertices = append(vertices,
		cubeVertices[i*3]*scale+pos.X,
		cubeVertices[i*3+1]*scale+pos.Y,
		cubeVertices[i*3+2]*scale+pos.Z,
	)
	colors = append(colors, color[0], color[1], color[2], color[3])
}

indices := cubeIndices

// Create and bind buffers
vertexBuffer := glctx.CreateBuffer()
glctx.BindBuffer(gl.ARRAY_BUFFER, vertexBuffer)
glctx.BufferData(gl.ARRAY_BUFFER, vertices, gl.STREAM_DRAW)

colorBuffer := glctx.CreateBuffer()
glctx.BindBuffer(gl.ARRAY_BUFFER, colorBuffer)
glctx.BufferData(gl.ARRAY_BUFFER, colors, gl.STREAM_DRAW)

indexBuffer := glctx.CreateBuffer()
glctx.BindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer)
glctx.BufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STREAM_DRAW)

// Set up vertex attribute
glctx.BindBuffer(gl.ARRAY_BUFFER, vertexBuffer)
glctx.VertexAttribPointer(r.vertexAttrib, 3, gl.FLOAT, false, 0, 0)

// Set up color attribute
glctx.BindBuffer(gl.ARRAY_BUFFER, colorBuffer)
glctx.VertexAttribPointer(r.colorAttrib, 4, gl.FLOAT, false, 0, 0)

// Compute MVP matrix = proj * view * model (model is identity with translation)
model := identity()
model[12] = pos.X
model[13] = pos.Y
model[14] = pos.Z

mvp := multiply(r.projMatrix, r.viewMatrix)
mvp = multiply(mvp, model)

glctx.UniformMatrix4fv(r.matrixUniform, mvp[:])

glctx.DrawElements(gl.TRIANGLES, len(indices), gl.UNSIGNED_SHORT, 0)

// Cleanup
glctx.DeleteBuffer(vertexBuffer)
glctx.DeleteBuffer(colorBuffer)
glctx.DeleteBuffer(indexBuffer)
 }

// ========================
// GAME INITIALIZATION
// ========================

func initGame() {
gameState.Player = Player{
Position: Vec3{0, 1, 0},
Yaw: 0,
Pitch: 0,
Health: 100,
Ammo: 30,
Alive: true,
}

// Spawn enemies randomly
for i := 0; i < MaxEnemies; i++ {
	gameState.Enemies[i] = Enemy{
		Position: Vec3{
			X: rand.Float32()*ArenaSize - ArenaSize/2,
			Y: 1,
			Z: rand.Float32()*ArenaSize - ArenaSize/2,
		},
		Health: 100,
		Alive:  true,
	}
}

for i := 0; i < MaxBullets; i++ {
	gameState.Bullets[i].Active = false
}

}

// ========================
// INPUT HANDLING
// ========================

var (
touchStartX float32
touchStartY float32
touchMoved bool
)

// Simple touch control:
// - Left half screen for movement (drag up/down/left/right)
// - Right half screen for aiming (drag to rotate camera)
// - Tap right half to shoot

func handleTouch(t touch.Event) {
switch t.Type {
case touch.TypeBegin:
touchStartX, touchStartY = t.X, t.Y
touchMoved = false
case touch.TypeMove:
dx := t.X - touchStartX
dy := t.Y - touchStartY
touchMoved = true

	if t.X < float32(renderer.width)/2 {
		// Movement control
		moveX := dx / 50
		moveY := -dy / 50
		updatePlayerMovement(moveX, moveY)
	} else {
		// Aiming control
		turnX := dx / 100
		turnY := dy / 100
		updatePlayerAim(turnX, turnY)
	}

	touchStartX, touchStartY = t.X, t.Y

case touch.TypeEnd:
	if !touchMoved && t.X >= float32(renderer.width)/2 {
		shootBullet()
	}
	stopPlayerMovement()
}

}

var moveDirection = Vec3{0, 0, 0}

func updatePlayerMovement(dx, dy float32) {
moveDirection = Vec3{dx, 0, dy}
}

func stopPlayerMovement() {
moveDirection = Vec3{0, 0, 0}
}

func updatePlayerAim(dx, dy float32) {
gameState.Player.Mutex.Lock()
defer gameState.Player.Mutex.Unlock()
gameState.Player.Yaw += dx
gameState.Player.Pitch -= dy
if gameState.Player.Pitch > 1.2 {
gameState.Player.Pitch = 1.2
}
if gameState.Player.Pitch < -1.2 {
gameState.Player.Pitch = -1.2
}
}

// ========================
// GAME LOGIC: SHOOTING
// ========================

func shootBullet() {
gameState.Player.Mutex.Lock()
defer gameState.Player.Mutex.Unlock()
now := time.Now()
if !gameState.Player.Alive || gameState.Player.Ammo <= 0 || now.Sub(gameState.Player.LastShot) < 300*time.Millisecond {
return
}
for i := range gameState.Bullets {
if !gameState.Bullets[i].Active {
gameState.Bullets[i].Active = true
gameState.Bullets[i].Position = gameState.Player.Position.Add(Vec3{0, 0.5, 0})
dir := getPlayerViewDirection()
gameState.Bullets[i].Velocity = dir.Mul(BulletSpeed)
gameState.Player.Ammo--
gameState.Player.LastShot = now
break
}
}
}

func getPlayerViewDirection() Vec3 {
p := &gameState.Player
cosPitch := float32(math.Cos(float64(p.Pitch)))
return Vec3{
X: float32(math.Sin(float64(p.Yaw))) * cosPitch,
Y: float32(math.Sin(float64(p.Pitch))),
Z: float32(math.Cos(float64(p.Yaw))) * cosPitch,
}.Normalize()
}

// ========================
// GAME LOGIC: UPDATE
// ========================

func updatePhysics(delta float32) {
// Update player movement
gameState.Player.Mutex.Lock()
dir := moveDirection
yaw := gameState.Player.Yaw
gameState.Player.Mutex.Unlock()

if dir.X != 0 || dir.Z != 0 {
	forward := Vec3{
		X: float32(math.Sin(float64(yaw))),
		Y: 0,
		Z: float32(math.Cos(float64(yaw))),
	}.Normalize()
	right := Vec3{-forward.Z, 0, forward.X}
	moveVec := forward.Mul(dir.Z).Add(right.Mul(dir.X)).Normalize().Mul(PlayerSpeed * delta)

	gameState.Player.Mutex.Lock()
	gameState.Player.Position = gameState.Player.Position.Add(moveVec)
	// Clamp player position inside arena
	if gameState.Player.Position.X > ArenaSize/2 {
		gameState.Player.Position.X = ArenaSize / 2
	}
	if gameState.Player.Position.X < -ArenaSize/2 {
		gameState.Player.Position.X = -ArenaSize / 2
	}
	if gameState.Player.Position.Z > ArenaSize/2 {
		gameState.Player.Position.Z = ArenaSize / 2
	}
	if gameState.Player.Position.Z < -ArenaSize/2 {
		gameState.Player.Position.Z = -ArenaSize / 2
	}
	gameState.Player.Mutex.Unlock()
}

// Update bullets
for i := range gameState.Bullets {
	b := &gameState.Bullets[i]
	if b.Active {
		b.Position = b.Position.Add(b.Velocity.Mul(delta))

		// Deactivate if out of arena
		if math.Abs(float64(b.Position.X)) > ArenaSize/2 || math.Abs(float64(b.Position.Z)) > ArenaSize/2 {
			b.Active = false
			continue
		}

		// Check collisions with enemies
		for j := range gameState.Enemies {
			enemy := &gameState.Enemies[j]
			if !enemy.Alive {
				continue
			}
			if distance(b.Position, enemy.Position) < 1.0 {
				enemy.Mutex.Lock()
				enemy.Health -= 25
				if enemy.Health <= 0 {
					enemy.Alive = false
				}
				enemy.Mutex.Unlock()
				b.Active = false
				break
			}
		}
	}
}

// Update enemies AI
for i := range gameState.Enemies {
	enemy := &gameState.Enemies[i]
	if enemy.Alive {
		enemyAI(enemy, delta)
	}
}

}

// Distance helper
func distance(a, b Vec3) float32 {
dx := a.X - b.X
dy := a.Y - b.Y
dz := a.Z - b.Z
return float32(math.Sqrt(float64(dxdx + dydy + dz*dz)))
}

// Simple enemy AI: chase player if within 15 units
func enemyAI(e *Enemy, delta float32) {
if !e.Alive {
return
}
dist := distance(e.Position, gameState.Player.Position)
if dist < 15 {
dir := gameState.Player.Position.Sub(e.Position).Normalize()
e.Position = e.Position.Add(dir.Mul(EnemySpeed * delta))
}
}

// ========================
// RENDERER STRUCT
// ========================

type Renderer struct {
glctx gl.Context
vertexAttrib uint32
colorAttrib uint32
matrixUniform gl.UniformLocation
projMatrix [16]float32
viewMatrix [16]float32
width int
height int
}

type Vec3 struct {
X, Y, Z float32
}

func (v Vec3) Add(o Vec3) Vec3 {
return Vec3{v.X + o.X, v.Y + o.Y, v.Z + o.Z}
}

func (v Vec3) Sub(o Vec3) Vec3 {
return Vec3{v.X - o.X, v.Y - o.Y, v.Z - o.Z}
}

func (v Vec3) Mul(s float32) Vec3 {
return Vec3{v.X * s, v.Y * s, v.Z * s}
}

func (v Vec3) Normalize() Vec3 {
len := float32(math.Sqrt(float64(v.Xv.X + v.Yv.Y + v.Z*v.Z)))
if len == 0 {
return Vec3{0, 0, 0}
}
return Vec3{v.X / len, v.Y / len, v.Z / len}
}

// Identity matrix helper
func identity() [16]float32 {
return [16]float32{
1, 0, 0, 0,
0, 1, 0, 0,
0, 0, 1, 0,
0, 0, 0, 1,
}
}

// Matrix multiplication helper (4x4)
func multiply(a, b [16]float32) [16]float32 {
var result [16]float32
for i := 0; i < 4; i++ {
for j := 0; j < 4; j++ {
sum := float32(0)
for k := 0; k < 4; k++ {
sum += a[i4+k] * b[k4+j]
}
result[i*4+j] = sum
}
}
return result
}
