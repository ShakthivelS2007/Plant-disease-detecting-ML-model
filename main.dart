import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:url_launcher/url_launcher.dart';

void main() {
  runApp(const MyApp());
}

// MAIN APP
class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  bool isDarkMode = false;

  void toggleTheme(bool value) {
    setState(() {
      isDarkMode = value;
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      themeMode: isDarkMode ? ThemeMode.dark : ThemeMode.light,
      theme: ThemeData(
        primarySwatch: Colors.green,
        scaffoldBackgroundColor: const Color(0xFFF1F8E9),
      ),
      darkTheme: ThemeData.dark(),
      home: SplashScreen(
        isDarkMode: isDarkMode,
        onToggle: toggleTheme,
      ),
    );
  }
}

// SPLASH SCREEN
class SplashScreen extends StatefulWidget {
  final bool isDarkMode;
  final Function(bool) onToggle;

  const SplashScreen({
    super.key,
    required this.isDarkMode,
    required this.onToggle,
  });

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController controller;

  @override
  void initState() {
    super.initState();

    controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    );

    WidgetsBinding.instance.addPostFrameCallback((_) {
      controller.forward();
    });

    Future.delayed(const Duration(seconds: 3), () {
      if (mounted) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (_) => HomeScreen(
              onToggle: widget.onToggle,
            ),
          ),
        );
      }
    });
  }

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor:
          widget.isDarkMode ? Colors.black : Colors.green,
      body: Center(
        child: ScaleTransition(
          scale: CurvedAnimation(
            parent: controller,
            curve: Curves.easeInOut,
          ),
          child: const Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.local_florist,
                  size: 90, color: Colors.white),
              SizedBox(height: 15),
              Text(
                "Plant Disease Identifier",
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// HOME SCREEN
class HomeScreen extends StatefulWidget {
  final Function(bool) onToggle;

  const HomeScreen({
    super.key,
    required this.onToggle,
  });

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  File? image;
  String result = "";
  bool loading = false;
  String? heatmapURL;
  bool showHeatmap = false;
  List<dynamic>? remedies;
  String? wikiURL;

  final picker = ImagePicker();

  Future<void> pickImage() async {
    final XFile? file =
        await picker.pickImage(source: ImageSource.camera);

    if (file != null) {
      setState(() {
        image = File(file.path);
        result = "";
      });

      sendImage(image!);
    }
  }

  Future<void> sendImage(File img) async {
    setState(() => loading = true);

    var request = http.MultipartRequest(
      'POST',
      Uri.parse("http://10.209.17.72:8000/predict"),
    );

    request.files.add(
      await http.MultipartFile.fromPath('file', img.path),
    );

    var response = await request.send();
    var data = await response.stream.bytesToString();
    var jsonData = json.decode(data);

    setState(() {
      heatmapURL = jsonData['heatmap'];
      showHeatmap = false;
      remedies = jsonData['remedies'];
      wikiURL = jsonData['wikipage'];
      result =
          "${jsonData["prediction"]}\nConfidence: ${jsonData["confidence"]}%";
      loading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("BioLens(Demo)"),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (_) => SettingsScreen(
                    onToggle: widget.onToggle,
                  ),
                ),
              );
            },
          )
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: SingleChildScrollView( 
          child: Column( 
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Container(
                height: 250,
                width: double.infinity,
                decoration: BoxDecoration(
                  color: Theme.of(context).brightness == Brightness.dark
                              ? Colors.grey[900]
                              : Colors.white,
                  borderRadius: BorderRadius.circular(16),
                  boxShadow: const [
                    BoxShadow(
                      color: Colors.black12,
                      blurRadius: 8,
                    )
                  ],
                ),
                child: image != null
                    ? ClipRRect(
                        borderRadius:
                            BorderRadius.circular(16),
                        child: Image.file(image!,
                            fit: BoxFit.cover),
                      )
                    : const Center(
                        child: Text("Capture a leaf image"),
                      ),
              ),
              const SizedBox(height: 20),
              SizedBox(
                width: double.infinity,
                height: 50,
                child: ElevatedButton.icon(
                  icon: const Icon(Icons.camera_alt),
                    label: const Text("Scan Leaf"),
                  onPressed: pickImage,
                ),
              ),

              // Result
              const SizedBox(height: 20),
              loading
                  ? const Center(
                      child: SizedBox(
                        height: 30,
                        width: 30,
                        child: CircularProgressIndicator(
                          strokeWidth: 3,
                        ),
                      ),
                    )
                  : Text(
                      result,
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    if (remedies != null)
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const SizedBox(height: 20),
                          const Text(
                            "Recommended Actions: ",
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(height: 10),
                          ...remedies!.map((remedy) => Padding(
                            padding: const EdgeInsets.symmetric(vertical: 4),
                            child: Row(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const Text(". "),
                                Expanded(child: Text(remedy)),
                              ],
                            ),
                          )),

                        ],
                      ),
                      if (wikiURL != null && wikiURL!.isNotEmpty)
                        Padding(
                          padding: const EdgeInsets.only(top: 20),
                          child: ElevatedButton.icon(
                            icon: const Icon(Icons.menu_book),
                            label: const Text("Learn More"),
                            onPressed: () async {
                              final Uri url = Uri.parse(wikiURL!);
                              await launchUrl(
                                url,
                                mode: LaunchMode.externalApplication,
                              );
                            },
                          ),
                        ),

                      //Heatmap
                      const SizedBox(height: 20),
                  
                      if (heatmapURL != null)
                        Column(
                          children: [
                            ElevatedButton.icon(
                              onPressed: () {
                                setState(() {
                                  showHeatmap = !showHeatmap;
                                });
                              },
                              icon: Icon(
                                showHeatmap ? Icons.visibility_off : Icons.visibility,
                              ),
                              label: Text(
                                showHeatmap ? "Hide Heatmap" : "Show Heatmap",
                              ),
                              style: ElevatedButton.styleFrom(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 20,
                                  vertical: 12,
                                ),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(30),
                                ),
                              ),
                            ),

                            const SizedBox(height: 15),

                            AnimatedSwitcher(
                              duration: const Duration(milliseconds: 400),
                              transitionBuilder: (child, animation) {
                                return FadeTransition(
                                  opacity: animation,
                                  child: SizeTransition(
                                    sizeFactor: animation,
                                    axisAlignment: -1,
                                    child: child,
                                  ),
                                );
                              },
                              child: showHeatmap
                                ? Container(
                                  key: const ValueKey("heatmap"),
                                  margin: const EdgeInsets.symmetric(horizontal: 20),
                                  decoration: BoxDecoration(
                                    borderRadius: BorderRadius.circular(20),
                                    boxShadow: [
                                      BoxShadow(
                                        color: Colors.black.withOpacity(0.2),
                                        blurRadius: 12,
                                        offset: const Offset(0, 6),
                                      ),
                                    ],
                                  ),
                                  child: ClipRRect(
                                    borderRadius: BorderRadius.circular(20),
                                    child: Image.network(
                                      heatmapURL!,
                                      height: 250,
                                      width: double.infinity,
                                      fit: BoxFit.cover,
                                      loadingBuilder: (context, child, progress) {
                                        if (progress == null) return child;
                                          return const SizedBox(
                                            height: 250,
                                            child: Center(
                                              child: CircularProgressIndicator(),
                                            ),
                                          );
                                      },
                                      errorBuilder: (context, error, stackTrace) {
                                        return const SizedBox(
                                          height: 250,
                                          child: Center(  
                                            child: Text("Failed to load heatmap"),
                                          ),
                                        );
                                      },
                                    ),
                                  ),
                                )
                                : const SizedBox(),
                            ),
                          ],
                        ),
            ],
          ),
        ),
      ),
    );
    
  }
}

// SETTINGS SCREEN
class SettingsScreen extends StatelessWidget {
  final Function(bool) onToggle;

  const SettingsScreen({
    super.key,
    required this.onToggle,
  });

  @override
  Widget build(BuildContext context) {
    final isDark =
        Theme.of(context).brightness == Brightness.dark;

    return Scaffold(
      appBar: AppBar(title: const Text("Settings")),
      body: ListTile(
        title: const Text("Dark Mode"),
        trailing: Switch(
          value: isDark,
          onChanged: onToggle,
        ),
      ),
    );
  }
}