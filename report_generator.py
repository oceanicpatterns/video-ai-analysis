import textwrap
from datetime import datetime
import statistics

def generate_txt_report(analysis_results):
    report_file = "swimming_analysis_report.txt"
    
    report = []
    report.append("Swimming Performance Analysis Report")
    report.append(f"Athlete: [Name]")
    report.append(f"Date: {datetime.now().strftime('%B %d, %Y')}")
    report.append("\nExecutive Summary:")
    
    all_body_angles = []
    for result in analysis_results.values():
        body_angles = [float(item.split()[6]) for item in result[1] if "body is at a" in item and item.split()[6].replace('.', '').isdigit()]
        all_body_angles.extend(body_angles)
    
    avg_body_angle = statistics.mean(all_body_angles) if all_body_angles else 0
    
    report.append(f"The analysis reveals an average body angle deviation of {avg_body_angle:.1f} degrees from horizontal across all strokes.")
    report.append("This indicates a need for focused improvement in maintaining a streamlined position.")
    
    report.append("\nStroke-specific Analysis:")
    
    for video_name, result in analysis_results.items():
        stroke, improvements, positive_feedback, equipment = result
        report.append(f"\n{stroke} (from {video_name}):")
        
        report.append("Areas for Improvement:")
        
        for item in improvements[:5]:  # Limit to top 5 improvements
            wrapped_text = textwrap.fill(item, width=70, initial_indent="- ", subsequent_indent="  ")
            report.append(wrapped_text)
        
        report.append("\nStrengths:")

        for item in positive_feedback[:5]:  # Limit to top 5 strengths
            wrapped_text = textwrap.fill(item, width=70, initial_indent="- ", subsequent_indent="  ")
            report.append(wrapped_text)
        
        report.append("--------------------------------------------------")
        
        if equipment:
            report.append("\nEquipment Detected:")
            for item in equipment:
                report.append(f"- {item}")
        
        report.append("-" * 50)
    
    report.append("\nOverall Recommendations:")
    recommendations = [
        "Focus on core strength and body awareness exercises to improve overall body alignment",
        "Practice stroke-specific drills to address individual technique issues",
        "Utilize underwater video analysis for visual feedback on body position",
        "Work with a coach on mental cues for maintaining proper alignment during swims"
    ]
    for i, rec in enumerate(recommendations, 1):
        report.append(f"{i}. {rec}")
    
    report.append("\nNext Steps:")
    next_steps = [
        "Develop a targeted training plan addressing key areas for improvement",
        "Schedule follow-up analysis in 3 months to assess progress",
        "Consider additional equipment like alignment boards or snorkels for technique work"
    ]
    for i, step in enumerate(next_steps, 1):
        report.append(f"{i}. {step}")
    
    try:
        with open(report_file, "w") as f:
            f.write("\n".join(report))
        print(f"Report generated successfully as '{report_file}'")
    except IOError as e:
        print(f"Error writing to file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return "\n".join(report)
