package main

import (
	"fmt"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type model struct {
	width  int
	height int
	tab    int
	progress int
	state  string
}

func (m model) Init() tea.Cmd { return nil }

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
	case tea.KeyMsg:
		switch msg.String() {
		case "q", "ctrl+c":
			return m, tea.Quit
		case "1", "2", "3", "4":
			m.tab = int(msg.Runes[0] - '1')
		case "r":
			m.state = "working"
			m.progress = 35
		}
	}
	return m, nil
}

var (
	activeStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("15")).Bold(true)
	mutedStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("244"))
	panelStyle  = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).Padding(1, 2)
	divStyle    = lipgloss.NewStyle().Margin(0, 1).Foreground(lipgloss.Color("238"))
)

func (m model) View() string {
	nav := lipgloss.NewStyle().Width(20).BorderRight(lipgloss.Border{Style: lipgloss.CharBorder, Left: "│"}).Render(
		lipgloss.JoinVertical(lipgloss.Left, "  1 Chat", "  2 Skills", "  3 Activity", "  4 Settings"),
	)
	header := lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("120")).Render("Symbio")
	bar := lipgloss.NewStyle().Border(lipgloss.NormalBorder()).Width(m.width-2).Render(
		fmt.Sprintf("State: %-8s  Progress %3d%%", m.state, m.progress),
	)
	content := panelStyle.Render("Primary workspace\nAdapter map: 5 sealed, 1 training, 0 pending")
	footer := "q quit • 1-4 tabs • r retry"
	main := lipgloss.JoinHorizontal(lipgloss.Top, nav, lipgloss.NewStyle().PaddingLeft(2).Render(content))
	return lipgloss.JoinVertical(lipgloss.Left, header, divStyle.Render(""), main, bar, footer)
}

func main() {
	tea.NewProgram(model{state: "idle", tab: 0}, tea.WithAltScreen()).Run()
}
